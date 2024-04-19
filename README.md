1. run the virtual environment through: source .venv/bin/activate
2. check the requirements: pip freeze
3. check qdrant images: docker images REPOSITORY
4. run qdrant: docker run -p 6333:6333 qdrant/qdrant
5. start uvicorn through this command: /Users/enaso/Library/CloudStorage/OneDrive-UniversiteitUtrecht/_GitHub/mira_gui/.venv/bin/python -m uvicorn main:app --reload
6. start working
