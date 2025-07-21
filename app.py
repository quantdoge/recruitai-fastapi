from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
import duckdb
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
from sqlalchemy.orm import sessionmaker
from urllib.parse import urlencode
from configparser import ConfigParser
from pinecone import Pinecone
from pinecone import ServerlessSpec
from pydantic import BaseModel, Field, field_validator, ValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import openai
from openai import OpenAI
from typing import List, Union, Optional, Dict
import time
from enum import Enum
import PyPDF2
from io import BytesIO
import dask
from dask import delayed
import json
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from data.users import users_db
from hashing import hash_string

config= ConfigParser()
config.read('config.ini')

## OPENAI CONNECTION ##
OPENAI_API_KEY= config.get('openai', 'API_KEY')
#OPENAI_API_KEY= 'failed' # test example
OPENAI_EMBEDDING_MODEL= config.get('openai', 'EMBEDDING_MODEL')

## DUCKDB CONNECTION ##
DUCKDB_KEY= config.get('duckdb', 'API_KEY')
#DUCKDB_KEY= 'failed' # test example
DUCKDB_URL= config.get('duckdb', 'URL')
DUCKDB_TABLE= config.get('duckdb', 'TABLE')

duckdb_param= {'motherduck_token':DUCKDB_KEY,
               'saas_mode':'false'}

duckdb_conn_url= DUCKDB_URL + '?' + urlencode(duckdb_param)

## POSTGRES CONNECTION ##
PG_HOST= config.get('pg', 'HOST')
PG_PORT= config.get('pg', 'PORT')
PG_DBNAME= config.get('pg', 'DB_NAME')
PG_USER= config.get('pg', 'USER')
PG_PWD= config.get('pg', 'PASSWORD')
#PG_PWD= 'failed' # test example

## FASTAPI SECURITY ##
SECRET_KEY= config.get('fastapi','SECRET_KEY')
ALGORITHM= config.get('fastapi','ALGORITHM')
ACCES_TOKEN_EXPIRE_MINUTES = 60
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

DATABASE_URL = f"postgresql://{PG_USER}:{PG_PWD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "sslmode": "require",
    }
)
SessionLocal = sessionmaker(bind=engine)


## PINECONE CONNECTION ##
PC_API_KEY= config.get('pinecone', 'SUMM_API_KEY')
PC_INDEX_NAME= config.get('pinecone', 'SUMM_INDEX_NAME')
PC_INDEX_NAME_REF= config.get('pinecone', 'INDEX_NAME')

pc = Pinecone(api_key=PC_API_KEY)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

BATCH_SIZE_JOB_RERANK= 10

# === FastAPI App ===
app = FastAPI(
    title="AI Recruiter API",  # <-- Change this to your desired title
    description="API documentation for the AI Recruiter backend server.",
    version="1.0.0"
)

token_auth_scheme = HTTPBearer()

def get_md_connection():
    conn = duckdb.connect(duckdb_conn_url)
    try:
        yield conn
    finally:
        conn.close()

def get_pg_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_openai_session():
    client = OpenAI(
		api_key=OPENAI_API_KEY
	)
    try:
        yield client
    finally:
        del client

def get_pc_session(index_name: str= Field(..., description="Pinecone index name")):
    index = pc.Index(index_name)
    try:
        yield index
    finally:
        del index

def get_pc_metadata(index_name: str= Field(..., description="Pinecone index name")):
	try:
		metadata = pc.describe_index(index_name)
		return 'success'
	except Exception as e:
		return str(e)

def get_pc_jd_metadata():
    return get_pc_metadata(PC_INDEX_NAME_REF)

def get_pc_summ_metadata():
    #return get_pc_metadata('failed') # test example
    return get_pc_metadata(PC_INDEX_NAME)

def get_pc_summ_session():
    #return get_pc_metadata('failed') # test example
    return get_pc_session(PC_INDEX_NAME)

def get_pc_ref_session():
    #return get_pc_metadata('failed') # test example
    return get_pc_session(PC_INDEX_NAME_REF)

def test_md_connection():
    try:
        md_conn_gen = get_md_connection()
        md_conn = next(md_conn_gen)
        md_result = md_conn.execute("SELECT 'Hello from MD'").fetchall()
        return 'success'
    except Exception as e:
        return str(e)

def test_pg_connection():
    try:
        pg_conn_gen = get_pg_session()
        pg_conn = next(pg_conn_gen)
        pg_result = pg_conn.execute(sqlalchemy.text("SELECT 'Hello from PG'")).fetchall()
        return 'success'
    except Exception as e:
        return str(e)

def test_openai_connection():
    try:
        openai_conn_gen = get_openai_session()
        openai_conn = next(openai_conn_gen)
        models = openai_conn.models.list()
        return 'success'
    except Exception as e:
        return str(e)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
	to_encode = data.copy()

	if expires_delta:
		expire = datetime.now() + expires_delta
	else:
		expire = datetime.now() + timedelta(minutes=ACCES_TOKEN_EXPIRE_MINUTES)

	to_encode.update({"exp": expire})
	encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
	return encoded_jwt

def get_current_user(token: str= Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
		status_code=401,
		detail="Could not validate credentials",
		headers={"WWW-Authenticate": "Bearer"},
	)

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

        return payload

    except JWTError:
        raise credentials_exception

class ModelEnum(str,Enum):
    text_embedding_3_small= 'text-embedding-3-small'
    gpt_4o_mini= 'gpt-4o-mini'
    gpt_4d1_mini= 'gpt-4.1-mini'

#### test: TESTING ALL AUTHENTICATIONS ####
class Test(BaseModel):
    motherduck: str = Field(..., description="MotherDuck connection status")
    postgresql: str = Field(..., description="PostgreSQL connection status")
    pinecone_jd: str = Field(..., description="Pinecone JD index metadata status")
    pinecone_summ: str = Field(..., description="Pinecone Summ index metadata status")
    openai_client: str = Field(..., description="OpenAI connection status")

@app.get("/test",description="A test to indicate if all backend connections are working", response_model=Test, tags=["Testing"])
async def sample_query(md_status=Depends(test_md_connection),
                 pg_status=Depends(test_pg_connection),
                 pc_jd= Depends(get_pc_jd_metadata),
                 pc_summ= Depends(get_pc_summ_metadata),
                 opneai_status=Depends(test_openai_connection)
                 ):

	if md_status=='success' and pg_status=='success' and pc_jd=='success' and pc_summ=='success' and opneai_status=='success':
		return JSONResponse(
        	status_code=200,
			content={
				"motherduck": md_status,
				"postgresql": pg_status,
				"pinecone_jd": pc_jd,
				"pinecone_summ": pc_summ,
                "openai_client": opneai_status
			}
		)
	else:
		return JSONResponse(
        	status_code=401,
			content={
				"motherduck": md_status,
				"postgresql": pg_status,
				"pinecone_jd": pc_jd,
				"pinecone_summ": pc_summ,
				"openai_client": opneai_status
			}
		)

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user= users_db.get(form_data.username)

    hashed_pwd= hash_string(form_data.password)

    if user is None or user.password !=hashed_pwd:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token= create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

class Input_embed(BaseModel):
    input_texts: List[str] = Field(..., description="List of input texts for embeddings")
    model: ModelEnum = Field(default=OPENAI_EMBEDDING_MODEL, description="OpenAI embedding model")

class Embed(BaseModel):
    embeddings: Union[List[List[float]], str] = Field(..., description="List of embeddings for the input texts")

@app.post("/embed", description="To embed a list of sentences", response_model= Embed, tags=["AIServices"])
async def get_embeddings(item: Input_embed,
                         username= Depends(get_current_user),
                         client=Depends(get_openai_session)):

    try:

        res = client.embeddings.create(
            input=item.input_texts, model=item.model
        )

        return JSONResponse(
			status_code=200,
			content={'embeddings':[record.embedding for record in res.data]}
		)

    except Exception as e:

        return JSONResponse(
             status_code=500,
             content={'error': str(e)}
		)

class Input_getTopMatches(BaseModel):
    input_texts: List[str] = Field(..., description="List of input texts for embeddings")
    model: ModelEnum = Field(default=OPENAI_EMBEDDING_MODEL, description="OpenAI embedding model")
    posting_date: int = Field(default= int(time.time()) - 60*60*24*14,
                              ge= int(time.time()) - 60*60*24*30,
                              le= int(time.time()),
                              description="Last Job Posting Datetime (epoch time in seconds)")
    minimum_salary: int = Field(default=None, description="Minimum salary to filter results",minimum=0,ge=0)
    top_k: int = Field(default= 100, description="Number of top matches to return",minimum=0,maximum=1000,ge=1,le=1000)
    selected_categories: str = Field(default= None, description="Comma-separated categories to filter results",min_length=7)

class GetTopMatches(BaseModel):
	similar_ids: dict = Field(..., description="Dictionary of similar IDs with their scores")

@app.post("/get-top-matches", description="To get the top N job ids based on cosine similarity", response_model= GetTopMatches, tags=["AIServices"])
async def get_top_ids(item: Input_getTopMatches,  # Changed from Input_embed
                      username= Depends(get_current_user),
                      client=Depends(get_openai_session),
                      index_gen=Depends(get_pc_summ_session)):
    try:
        query_embedding = client.embeddings.create(
            input=item.input_texts, model=item.model
        )

        # Define filter variables (replace these with actual logic or parameters as needed)
        if item.selected_categories is not None:
            FILTERED_CATEGORIES = item.selected_categories.split(',')
        else:
            FILTERED_CATEGORIES = None

        FILTERED_TIME = item.posting_date
        FILTERED_MIN_SALARY = item.minimum_salary
        INDEX_SUMM_K = item.top_k

        conditions = []

        conditions.append({"newPostingDate": {"$gte": FILTERED_TIME}})

        if FILTERED_MIN_SALARY is not None:
            conditions.append({"maximum": {"$gte": FILTERED_MIN_SALARY}})

        if FILTERED_CATEGORIES is not None:
            conditions.append({"categories_ds": {"$in": FILTERED_CATEGORIES}})

        #filter_conditions = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        filter_conditions = {"$and": conditions}

        index= next(index_gen)  # Get the Pinecone index from the generator

        query_result = index.query(
            vector=query_embedding.data[0].embedding,  # Fixed: get actual embedding vector
            filter=filter_conditions,
            top_k=INDEX_SUMM_K,
            include_metadata=True
        )

        # Extract IDs from metadata
        scoring_matrix = {}
        for match in query_result.matches:
            id = match.metadata.get('id')
            if scoring_matrix.get(id, None) is None:
                scoring_matrix[id] = match.score  # Fixed: use .score instead of .get('score')
            else:
                scoring_matrix[id] = max(match.score, scoring_matrix[id])

        scoring_matrix_sorted = dict(sorted(scoring_matrix.items(), key=lambda x: x[1], reverse=True))

        return JSONResponse(
            status_code=200,
            content={'similar_ids': scoring_matrix_sorted}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

class Input_rebalaTopMatches(BaseModel):
	top_ids_interests: Dict[str,float] = Field(..., description="Dictionary of top matching Job IDs based on interests")
	top_ids_profile: Dict[str,float]  = Field(..., description="Dictionary of top matching Job IDs based on job experience and skills")
	profile_weight: float = Field(default=0.5, ge=0, le=1, description="Weight for model to consider based on job experience and skills, in reciprocal to weight for interests")
	index_summ_k: int = Field(default=100, ge=1, le=1000, description="Number of top matches to return")

class RebalTopMatches(BaseModel):
	top_ids: List[str] = Field(..., description="List of top matching Job IDs after rebalancing interests and experiences")

@app.post("/rebal-top-matches",description="Rebalance top searches from interests and experiences ",
          response_model=RebalTopMatches ,tags=["AIServices"])
async def rebelance_top_matches(item: Input_rebalaTopMatches,username= Depends(get_current_user)):

	try:
		top_ids={}
		interests_weight= 1- item.profile_weight
		profile_weight= item.profile_weight

		for uuid, weight in item.top_ids_interests.items():
			top_ids[uuid]=interests_weight*weight

		for uuid, weight in item.top_ids_profile.items():

			if top_ids.get(uuid,None) is None:
				top_ids[uuid]=profile_weight*weight
			else:
				top_ids[uuid]=profile_weight*weight + top_ids[uuid]

		top_ids_sorted= sorted(top_ids_sorted.items(), key=lambda x: x[1], reverse=True)
		top_ids_flatten= [x[0] for x in top_ids_sorted]
		filered_top_ids_flatten= top_ids_flatten[:item.index_summ_k]

		return JSONResponse(
			status_code=200,
			content={'top_ids': filered_top_ids_flatten}
		)

	except Exception as e:
		return JSONResponse(
			status_code=500,
			content={'error': str(e)}
		)

class UploadPDF(BaseModel):
	filename: str = Field(..., description="Name of the uploaded PDF file")
	size: int = Field(..., description="Size of the extracted text content in characters")
	details: str = Field(..., description="Extracted text content from the PDF")

@app.post("/pdf-to-text", description="Upload a PDF file and extract its text content",
          response_model=UploadPDF,
          tags=["Data Load"])
async def upload_pdf(file: UploadFile = File(...),username= Depends(get_current_user)):
    # File type validation
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Optional: Further check by filename extension (not strictly required)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must have a .pdf extension.")

    content = await file.read()
    pdf_buffer = BytesIO(content)

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_buffer)

        extracted_text = ""

        for page in pdf_reader.pages:
            text = page.extract_text()

            if text:
                extracted_text += text

        # Set a threshold: e.g., at least 100 characters of text required
        min_text_length = 7

        if len(extracted_text.strip()) < min_text_length:
            raise HTTPException(
                status_code=400,
                detail="This PDF does not contain extractable text. It may be a scanned image PDF (not a digital/text-based PDF)."
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF processing error: {str(e)}")

    return JSONResponse(status_code=200,
                        content={"filename": file.filename,
                                 "size": len(extracted_text.strip()),
                                 "details": extracted_text.strip()})

class Input_prepareAssistantPrompt(BaseModel):
    job_ids: List[str] = Field(..., description="List of job IDs to filter metadata and embeddings")

class PrepareAssistantPrompt(BaseModel):
	prompt: List[str] = Field(..., description="List of formatted prompts for the AI assistant based on job metadata and embeddings")

@app.post("/prepare-assistant-prompt",
          description="Prepare a prompt for the AI assistant based on job metadata and embeddings",
          response_model=PrepareAssistantPrompt,
		  tags=["Data Load"])
async def prepare_assistant_prompt(item: Input_prepareAssistantPrompt,
                                   username= Depends(get_current_user),
							       con = Depends(get_md_connection),
                                   pc_index_ref= Depends(get_pc_ref_session)):

    try:

        formatted_ids = ','.join(f"'{v}'" for v in item.job_ids)
        result = con.execute(f'SELECT * FROM {DUCKDB_TABLE} WHERE uuid IN [{formatted_ids}]').fetchall()
        datafields = [col[0] for col in con.description]
        md_dicts = [dict(zip(datafields, row)) for row in result]
        md_dict_rearranged = {d['uuid']: d for d in md_dicts}

        output_prompt= 'Following are the jobs available:\n\n'
        job_uuids= list(md_dict_rearranged.keys())

        ref_index= next(pc_index_ref)
        jobs_embeddings= ref_index.fetch(ids=item.job_ids)

        output_prompt_master= []

        for idx in job_uuids:
            output_prompt='#### START OF JOB ID: {job_id}####\n'.format(job_id=idx)
            output_prompt+='Metadata of this job:{jobs_metadata_str}\n'.format(jobs_metadata_str=str(md_dict_rearranged[idx]))

            try:
                output_prompt += 'Description of this job:{jobs_descr_str}\n'.format(
                    jobs_descr_str=jobs_embeddings.vectors[idx]['metadata']['text']
                )
            except Exception:
                pass

            output_prompt+='#### END OF JOB ID: {job_id}####\n\n'.format(job_id=idx)

            output_prompt_master.append(output_prompt)

        return JSONResponse(
			status_code=200,
			content={'prompt': output_prompt_master}
		)

    except Exception as e:
        return JSONResponse(
			status_code=500,
			content={'error': str(e)}
		)

class Input_jobRerank(BaseModel):
    model: ModelEnum = Field(default='gpt-4.1-mini', description='LLM Model for Reranking')
    user_query_interest: str = Field(..., description="User interests for job matching", min_length=30)
    user_profile: str = Field(..., description="User career profile including experiences and qualifications", min_length=30)
    system_prompt: str = Field(
        default=(
            "You are a professional recruiter. You are given a list of jobs and candidates career interests and career profile. "
            "Your task is to evaluate and rerank these jobs for the candidates based on their interest match and job fit against the job descriptions given. "
            "And also suggest summary and objectives for the candidate to put in their CVs in order to improve their job search and career development."
        ),
        description='System Prompt for Job Reranking',
        min_length=30
    )
    job_id_assistant_prompt: List[str] = Field(..., description="Assistant Prompt on Job ID generated from prepare-assistant-prompt services",max_length=10,min_length=1)
    temperature: float = Field(default=0.75, ge=0, le=1, description="Temperature for the LLM model to generate responses")
    max_tokens: int = Field(default=20000, ge=1000, le=20000, description="Maximum number of tokens for the LLM model to generate responses")

class output_job_obj(BaseModel):
    id: str= Field(..., description="Job ID")
    interest_match: List[str]= Field(description="List of reasons why you think this job matches the candidate interest")
    interest_unmatch: List[str]= Field(description="List of reasons why you think this job does not match the candidate interest")
    job_fit: List[str]= Field(description="List of reasons why you think this candidate qualified for this job")
    job_unfit: List[str]= Field(description="List of reasons why you think this candidate not qualified for this job and should work mroe on")
    relevancy: float= Field(..., description="Relevant score of this job to the candidate interest",min=0,max=100)
    success_prob: float= Field(..., description="Estimated success probability of this job to the candidate experience and qualifications",min=0,max=100)
    suggested_cv_summary: str= Field(..., description="Suggested summary for the candidate to put in their CV")
    suggested_cv_objectives: str= Field(..., description="Suggested objectives for the candidate to put in their CV")

class output_job_list(BaseModel):
     jobs: List[output_job_obj]= Field(..., description="List of jobs that are most relevant to the candidate query")

output_job_list_schema= str(output_job_list.model_json_schema())

def chunk_list(lst, chunk_size):
    """Split a list into chunks of specified size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

class JobRerank(BaseModel):
    jobs: List[output_job_obj] = Field(..., description="List of jobs ranked by relevancy")
    total_jobs_processed: int = Field(..., description="Total number of jobs processed")
    total_batches: int = Field(..., description="Number of batches processed")
    errors: Optional[List[dict]] = Field(None, description="List of errors if any occurred")


@app.post("/job-rerank", tags=["AIServices"],
          response_model=output_job_list,
          description="Rerank jobs based on candidate interests, profile and job descriptions",)
async def job_rerank(item: Input_jobRerank,
                     username= Depends(get_current_user),
                     client=Depends(get_openai_session)):

    try:
        # Combine all prompts in the batch
        combined_prompt = "\n\n".join(item.job_id_assistant_prompt)
        model_param = item.model

        response = client.chat.completions.create(
            model=model_param, #ModelEnum['model_param'],
            messages=[
                {"role": "system", "content": item.system_prompt},
                {"role": "user", "content": "Candidate Interests: \n" + item.user_query_interest},
                {"role": "user", "content": "Candidate Profile (Job Experiences and Qualifications): \n " + item.user_profile},
                {"role": "assistant", "content": combined_prompt},
                {"role": "user", "content": "Please output your answer in the JSON format of the following schema: " + output_job_list_schema}
            ],
            response_format={"type": "json_object"},
            temperature=item.temperature,
            max_tokens=item.max_tokens
        )

        content = response.choices[0].message.content
        return JSONResponse(
            status_code=200,
            content=json.loads(content)
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

class Input_getJobMd(BaseModel):
    job_ids: List[str] = Field(..., description="List of job UUIDs to retrieve metadata",min_length=1)

class JobMetadata(BaseModel):
    uuid: str = Field(..., description="Job UUID")
    title: str = Field(..., description="Job title")
    name: str = Field(..., description="Company name")
    minimum: Optional[int] = Field(None, description="Minimum salary")
    maximum: Optional[int] = Field(None, description="Maximum salary")
    categories_ds: Optional[str] = Field(None, description="Job categories")
    jobDetailsUrl: Optional[str] = Field(None, description="URL to job posting page")

class GetJobMd(BaseModel):
    jobs: List[JobMetadata] = Field(..., description="List of job metadata")

@app.post("/get-job-md",
          description="Get job metadata by list of job UUIDs",
          response_model=GetJobMd,
          tags=["Data Load"])

async def get_job_metadata(item: Input_getJobMd, username= Depends(get_current_user),con=Depends(get_md_connection)):
    try:
        # Format the job IDs for SQL query
        formatted_ids = ','.join(f"'{job_id}'" for job_id in item.job_ids)

        # Execute query to get specific fields
        query = f"""
        SELECT uuid, title, name, minimum, maximum, categories_ds, jobDetailsUrl
        FROM {DUCKDB_TABLE}
        WHERE uuid IN ({formatted_ids})
        """

        result = con.execute(query).fetchall()

        # Get column names
        columns = ['uuid', 'title', 'name', 'minimum', 'maximum', 'categories_ds', 'jobDetailsUrl']

        # Convert results to list of dictionaries
        jobs_data = []
        for row in result:
            job_dict = dict(zip(columns, row))
            jobs_data.append(job_dict)

        return JSONResponse(
            status_code=200,
            content={
                'jobs': jobs_data
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

