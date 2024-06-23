## Steps to correctly install the dependent libraries on local machine
1. `git clone https://github.com/yotamdery/llm_for_qa`
2. `python3.12 -m venv .venv`
(That's the python version used for this project)
3. `source .venv/bin/activate`
4. `pip install --upgrade pip`
5. `pip install -r requirements.txt`

6. For convenience (not mandatory) - create .vscode/settings.json with the following:
{
    "python.terminal.activateEnvironment": true,
    "python.defaultInterpreterPath": ".venv/bin/python3"
}
