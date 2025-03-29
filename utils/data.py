import os
import pandas as pd


class Stackoverflowloader:
  def __init__(self, file_path=None):

      if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cleaned_data.csv')
      self.df = pd.read_csv(file_path)

  
  def load(self):
    return self.df

  def describe_candidate(self, candidate_row):
    can= candidate_row.to_dict()

    # Process individual fields

    age= 'under 35' if can['Age']== '<35' else 'over 35'
    edu=  'Masters' if can['EdLevel'] == 'Master' else can['EdLevel']
    gender = can['Gender']
    country=can['Country']
    worked_with =','.join(can['HaveWorkedWith'].split(';'))
    curr_employed='Currently employed' if can['Employed'] ==1 else 'currently unemployed'

           
    # Return the formatted string
    return (
      f"Candidate is {age} years old. "
      f"Candidate has a {edu} degree. "
      f"Candidate is a {gender}. "
      f"Candidate is from the country of {country}. "
      f"Candidate has worked with {worked_with}. "
      f"Candidate is {curr_employed}."
      )

# The main execution block

if __name__ =='__main__':

  loader= Stackoverflowloader()
  df= loader.load()

  df['Description']= df.apply(lambda row: loader.describe_candidate(row),axis=1)

  df.to_csv('../data/cleaned_data.csv', index=False)

  print(df[['Description']].head())