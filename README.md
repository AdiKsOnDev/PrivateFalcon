# 🦅 PrivateFalcon
PrivateFalcon is a Python script that allows you to locally query documents using the Falcon-7b Language Model (L.L.M) from HuggingFace. This script is designed to work with documents that have been ingested into a VectorStore using the `ingest.py` file. With PrivateFalcon, you can perform efficient and accurate document retrieval and similarity searches.

# 🔧 Prerequisites
Before you begin, make sure you have the following prerequisites in place:

- Python 3.x
- A pre-trained Falcon-7b model (Will be installed by the script)
- Documents Placed in the `data/` directory
- .env file containing:
```bash
DB_DIRECTORY=<directory-name>
EMBEDDINGS_MODEL=all-MiniLM-L6-v2 
```
 You can put any other embeddings model into that variable.

# 📥 Installation

1. Clone the repository
```bash
git repo clone https://github.com/AdiKsOnDev/PrivateFalcon.git
```

2. Install the dependencies
```bash
pip install -r requirements
```

# 📊 Usage
PrivateFalcon is easy to use:

1. Place your documents into the `data/` directory.
2. Run:
```bash
python ingest.py
```
3. After creating a VectorStore, run:
```bash
python main.py
```

# 📂 Directory Structure
```
PrivateFalcon/
├── main.py             # Ask questions
├── ingest.py           # Script that ingests your documents
├── vectors/            # Directory with ingested documents
├── data/               # Directory with the source documents
├── requirements.txt    # .txt file with all the dependencies
```
##

# 🤝 Contributing
If you want to contribute to PrivateFalcon, feel free to submit a pull request or make an Issue

# 📧 Contact
For any questions or issues, please contact me at adilalizade13@gmail.com

Happy querying with PrivateFalcon! 🦅🔍