# Basic Eagle Framework Example

This example demonstrates the core features of the Eagle framework, including:

- Database models with SQLAlchemy
- User authentication with JWT
- Admin interface
- Basic CRUD operations

## Prerequisites

- Python 3.8+
- pip

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/eagle.git
   cd eagle/examples/basic_app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration.

## Running the Example

Start the development server:
```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Welcome message
- `GET /docs`: Interactive API documentation (Swagger UI)
- `GET /redoc`: Alternative API documentation (ReDoc)
- `POST /token`: Get access token (username: admin, password: admin)
- `GET /users/me`: Get current user info (requires authentication)
- `POST /users/`: Create a new user

## Admin Interface

Access the admin interface at `http://localhost:8000/admin`

Default admin credentials:
- Email: admin@example.com
- Password: admin

## Project Structure

```
basic_app/
├── main.py           # Main application module
├── requirements.txt  # Dependencies
└── README.md        # This file
```
