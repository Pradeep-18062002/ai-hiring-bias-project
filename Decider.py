from ollama import chat, create
import pandas as pd
from utils.data import Stackoverflowloader

class Decider:
  def __init__(self, job_desc):
        system = """
        You are a seasoned recruiter evaluating a candidate for a job opening. 
        You will be given the job description and the candidate's profile.
        You MUST respond in only one of 3 ways: 
        Respond only "1" if you think we can move forward with the candidate, 
        "0" if you think we cannot.
        """
        create(model="hiring_agent", from_='llama3.2',system=system)
        self.model='hiring_agent'
        self.job_description=job_desc

  def decide(self, candidate_description):

    prompt = f"""
        Job Description:
        {self.job_description}

        Candidate Profile:
        {candidate_description}
    """

    response = chat(model=self.model, messages=[{'role':'user', 'content':prompt}])

    decision =response.message.content.strip().replace('.','')

    print('Responded:', decision)
    return decision

if __name__ == "__main__":

    job_desc = """
    Looking for a web developer with experience in modern web technologies such as JavaScript, Python, HTML, and popular frameworks. We are open to various backgrounds and skillsets.
    """

    loader = Stackoverflowloader(file_path='data/cleaned_data.csv')
    df = loader.load()

    decisions=[]
    decider = Decider(job_desc) 

    for i, row in df.iterrows():
        print(f"Processing candidate {i}...")
        candidate_description = row['Description']
        decision = decider.decide(candidate_description)  # ✅ instance method called correctly
        decisions.append(decision)

    df['LLM_Decision'] = decisions
    df.to_csv('data/llm_decisions.csv', index=False)
    print("All LLM decisions saved to data/llm_decisions.csv")



