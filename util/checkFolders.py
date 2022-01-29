from pathlib import Path

def checkFolders(folders):
	for f in folders:
		Path(f).mkdir( parents=True,
						exist_ok=True)