NLP Sentiment Analysis API

Project Description
This project is a secure and high-performance Natural Language Processing API built using FastAPI and Hugging Face Transformers. It analyzes the sentiment of input text and returns structured results whether the text is positive or negative.

The system uses a locally deployed pre-trained model, removing the need for external cloud services and reducing operational costs.


Features
- Sentiment Analysis using a pre-trained NLP model
- Secure API access via Bearer Token Authentication
- Rate Limiting (5 requests per minute per user)
- Fast and efficient local processing
- Automatic API documentation (Swagger UI & ReDoc)


 Tech Stack
- FastAPI (Backend Framework)
- Hugging Face Transformers (NLP Library)
- DistilBERT Model (Pre-trained sentiment model)
- SlowAPI (Rate Limiting)
- Python 3.10+


 Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Internet connection (for first-time model download)


 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/aysha312/SIA.git
cd SIA
