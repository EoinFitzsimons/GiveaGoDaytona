import os

from dotenv import load_dotenv
from daytona import Daytona, DaytonaConfig
load_dotenv()
config = DaytonaConfig(api_key=os.getenv("DAYTONA_API_KEY"))
daytona = Daytona(config)
sandbox = daytona.create()
response = sandbox.process.code_run('print("Hello World")')
print(response.result)