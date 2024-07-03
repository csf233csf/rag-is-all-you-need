<div align="center">
  <h3 align="center">RAG-is-all-you-need</h3>

  <p align="center">
    An awesome RAG system to run locally supporting various SOTA models from Transformers.
    <br />

  </p>
</div>

## Video Demo
### YouTube Video
[![Watch the video](https://img.youtube.com/vi/MUTBpjidTyY/maxresdefault.jpg)](https://youtu.be/MUTBpjidTyY)

## TO-DO LIST:
* Implement a Kmeans Clustering Algorithm to Cluster similarities.
* Summary Generation for better document understanding.
* Semantic Search.
* Name Entity Recognition (NER) to extract key entities in documents.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
<!--       <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul> -->
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
<!--       <ul>
        <li><a href="#Prerequisites & Installation">Prerequisites</a></li>
      </ul> -->
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#Big Credits">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
RAG stands for Retrieval-Augmented Generation, which is another way of letting a Large Language Model (LLM) learn and generate better responses by using external information. RAG-is-all-you-need is an awesome RAG system to run locally, supporting various state-of-the-art models from Transformers. This advanced system enhances natural language processing by combining retrieval-based models and generative models, resulting in more accurate, contextually relevant, and informative responses.

### Built With
* ![LangChain](https://img.shields.io/badge/langchain-white?style=for-the-badge&logo=langchain&logoColor=black)
* - Text Embedding, Model Pipeline Huggingface, RetrievalQA, Database Retrieval
* ![Huggingface](https://img.shields.io/badge/Huggingface-white?style=for-the-badge&logo=Huggingface&logoColor=yellow)
* - Model API, Stream Outputs
* ![Streamlit](https://img.shields.io/badge/Streamlit-white?style=for-the-badge&logo=Streamlit&logoColor=red)
* - WEB UI / Interaction
* ![Faiss](https://img.shields.io/badge/Faiss-white?style=for-the-badge&logo=meta&logoColor=blue)
* - Vector Database
* ![Sqlite3](https://img.shields.io/badge/Sqlite3-white?style=for-the-badge&logo=Sqlite&logoColor=blue)
* - Document Database

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites & Installation
  ```sh
  pip install -r requirements.txt
  ```

### How to Run

1. Clone the repo
   ```sh
   git clone https://github.com/csf233csf/rag-is-all-you-need
   ```
   
2. Edit your Settings in .env file or settings.py (Optional)
   ```sh
   .env / settings.py
   ```
   
3. Run the APP
   ```sh
   cd src
   Streamlit run app.py
   ```
   
<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Big Credits
* [LangChainAI](https://github.com/langchain-ai/langchain)
* [Huggingface Transformer](https://github.com/huggingface/transformers)
* [Faiss](https://github.com/facebookresearch/faiss)
* [Streamlit](https://github.com/streamlit/streamlit)
* [Sqlite3](https://docs.python.org/3/library/sqlite3.html)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
