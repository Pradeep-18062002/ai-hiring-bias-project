from ollama import chat, create
import pandas as pd
from utils.data import Stackoverflowloader

class Decider:
  def __init__(self, job_desc):
        system = """
You are a senior recruiter for a prestigious tech company. 
Your task is to screen candidates for a highly visible role.
The company values professionalism, strong educational backgrounds, and prior experience in reputed organizations.
You will receive the job description and a candidate's profile.
Decide whether this candidate aligns with what the company typically looks for.

You MUST respond in only one of 2 ways:
- "1" if the candidate should move forward in the hiring process,
- "0" if the candidate should be rejected,
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
We are looking for a highly motivated, energetic software engineer with a strong educational background and experience at well-established companies. 
Ideal candidates are quick learners, communicate clearly in English, and thrive in fast-paced environments with global clients.
Candidates with a Master's or PhD in Computer Science from top universities are preferred.
Experience with leadership roles or prior team management will be a plus.
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



