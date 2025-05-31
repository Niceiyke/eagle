# Eagle Admin Dashboard

The Eagle Admin Dashboard provides an automatic admin interface for managing your application's database models with full CRUD (Create, Read, Update, Delete) capabilities.

## Features

- Auto-discovers SQLAlchemy models
- Automatic form generation based on model fields
- List, create, edit, and delete records
- Responsive design that works on all devices
- Built with Tailwind CSS for easy theming

## Installation

1. Make sure you have Eagle API installed and set up in your project.

2. Import and initialize the admin in your main application file:

```python
from eagleapi.admin import setup_admin

# Initialize your app
app = EagleAPI()

# Set up the admin interface
admin = setup_admin(
    app, 
    path="/admin",  # Customize the admin URL path
    models_module="your_app.models"  # Path to your models module
)
```

## Usage

1. Start your application and navigate to `/admin` (or your custom path).
2. You'll see a list of all registered database models.
3. Click on a model to view, create, edit, or delete records.

## Customization

### Model Registration

By default, the admin will automatically discover models from the specified `models_module`. You can also manually register models:

```python
from eagleapi.admin import model_registry
from your_app.models import YourModel

# Register a model
model_registry.register(YourModel)
```

### Field Types

The admin supports various field types including:
- String/Text
- Integer/Float
- Boolean
- DateTime
- ForeignKey (displayed as dropdowns)

### Field Options

You can customize field behavior using SQLAlchemy column attributes:
- `nullable`: Makes the field optional
- `default`: Sets a default value
- `primary_key`: Identifies the primary key

## Security

Make sure to implement proper authentication and authorization before using the admin interface in production. The admin interface currently doesn't include built-in authentication.

## Styling

The admin interface uses Tailwind CSS. You can customize the look and feel by:

1. Overriding the templates in your project
2. Adding custom CSS classes
3. Extending the base template

## Requirements

- Python 3.7+
- FastAPI
- SQLAlchemy
- Pydantic
- Jinja2
- Tailwind CSS (included via CDN)

## License

This project is licensed under the MIT License.
