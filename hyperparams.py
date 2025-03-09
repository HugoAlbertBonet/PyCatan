import pandas as pd
from ga import GA
from tqdm import tqdm
from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta
import time
import boto3
from io import StringIO

session = boto3.Session()
client = session.client('s3')

def save_dataframe_to_s3(df, bucket_name, file_name):
    """Saves a Pandas DataFrame to an S3 bucket as a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        bucket_name (str): The name of the S3 bucket.
        file_name (str): The desired file name in the bucket.
    """
    try:
        # Convert DataFrame to CSV string
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)

        # Upload CSV string to S3
        client.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())

        print(f"DataFrame successfully saved to s3://{bucket_name}/{file_name}")
    except Exception as e:
        print(f"Error saving DataFrame to S3: {e}")



list_hyperparams = []
dic_hyperparams = {
    "pop_size": [],
    "generations": [],
    "rounds": [],
    "tournament_size": [],
    "fitness": [],
    "time": []
}

AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
IND_SIZE = len(AGENTS)

for pop_size in [20,10,5]:
    for generations in [20,10,5]:
        for rounds in [20,10,5]:
            for tournament_size in [2,1]:
                list_hyperparams.append((pop_size, generations, rounds, tournament_size))

for pop_size, generations, rounds, tournament_size in tqdm(list_hyperparams):
    ga = GA(IND_SIZE, AGENTS, rounds = rounds)
    t1 = time.time()
    ga(pop_size = pop_size, generations = generations, tournament_size = tournament_size)
    t2 = time.time()
    dic_hyperparams["pop_size"].append(pop_size)
    dic_hyperparams["generations"].append(generations)
    dic_hyperparams["rounds"].append(rounds)
    dic_hyperparams["tournament_size"].append(tournament_size)
    dic_hyperparams["fitness"].append(ga.min_fit)
    dic_hyperparams["time"].append(t2-t1)

df = pd.DataFrame(dic_hyperparams)

save_dataframe_to_s3(df, "pycatan", "ga_results.csv")


