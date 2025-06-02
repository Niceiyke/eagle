# examples/models.py
"""
Example models using the enhanced database setup
"""
from your_db_module import BaseModel, Mapped, mapped_column, String, Boolean, Text, ForeignKey, Integer
from sqlalchemy.orm import relationship


class User(BaseModel):
    """User model with auto-generated schemas"""
    __tablename__ = "users"
    
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(50), nullable=False)
    last_name: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")

class Post(BaseModel):
    """Post model with auto-generated schemas"""
    __tablename__ = "posts"
    
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False)
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Relationships
    author = relationship("User", back_populates="posts")

# examples/main.py
"""
FastAPI application using the enhanced database
"""
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from your_db_module import get_db, create_all_tables, create_crud_routes, inspect_model_schemas
from models import User, Post

app = FastAPI(title="Blog API with Auto-Generated Schemas")

@app.on_event("startup")
async def startup():
    """Initialize database and inspect schemas"""
    print("Creating database tables...")
    await create_all_tables()
    
    print("Inspecting generated schemas...")
    inspect_model_schemas(User)
    inspect_model_schemas(Post)

# Method 1: Auto-generate full CRUD routes
app.include_router(create_crud_routes(User, "/users"))
app.include_router(create_crud_routes(Post, "/posts"))

# Method 2: Manual routes with auto-generated schemas
@app.post("/users/register", response_model=User.get_response_schema())
async def register_user(
    user_data: User.get_create_schema(),
    session: AsyncSession = Depends(get_db)
):
    """Custom user registration with validation"""
    # Check if user already exists
    existing_user = await User.get_by(session, username=user_data.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    existing_email = await User.get_by(session, email=user_data.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create new user
    try:
        user = await User.create(session, **user_data.model_dump())
        await session.commit()
        return user
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/{user_id}/posts", response_model=list[Post.get_list_schema()])
async def get_user_posts(
    user_id: int,
    session: AsyncSession = Depends(get_db)
):
    """Get all posts by a specific user"""
    user = await User.get(session, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    posts = await Post.get_all(session, author_id=user_id)
    return posts

@app.post("/users/{user_id}/posts", response_model=Post.get_response_schema())
async def create_user_post(
    user_id: int,
    post_data: Post.get_create_schema(),
    session: AsyncSession = Depends(get_db)
):
    """Create a new post for a user"""
    user = await User.get(session, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Add author_id to post data
        post_dict = post_data.model_dump()
        post_dict['author_id'] = user_id
        
        post = await Post.create(session, **post_dict)
        await session.commit()
        return post
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))

# Method 3: Using schemas for validation in custom business logic
@app.put("/users/{user_id}/profile", response_model=User.get_response_schema())
async def update_user_profile(
    user_id: int,
    profile_data: User.get_update_schema(),
    session: AsyncSession = Depends(get_db)
):
    """Update user profile with custom validation"""
    user = await User.get(session, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Custom validation
    update_data = profile_data.model_dump(exclude_unset=True)
    
    # Check if email is being changed and if it's unique
    if 'email' in update_data:
        existing_email = await User.get_by(session, email=update_data['email'])
        if existing_email and existing_email.id != user_id:
            raise HTTPException(status_code=400, detail="Email already in use")
    
    # Check if username is being changed and if it's unique
    if 'username' in update_data:
        existing_username = await User.get_by(session, username=update_data['username'])
        if existing_username and existing_username.id != user_id:
            raise HTTPException(status_code=400, detail="Username already in use")
    
    try:
        await user.update(session, **update_data)
        await session.commit()
        return user
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))

# Method 4: Bulk operations with schemas
@app.post("/users/bulk", response_model=list[User.get_response_schema()])
async def create_bulk_users(
    users_data: list[User.get_create_schema()],
    session: AsyncSession = Depends(get_db)
):
    """Create multiple users at once"""
    created_users = []
    
    try:
        for user_data in users_data:
            # Check for duplicates
            existing = await User.get_by(session, username=user_data.username)
            if existing:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Username '{user_data.username}' already exists"
                )
            
            user = await User.create(session, **user_data.model_dump())
            created_users.append(user)
        
        await session.commit()
        return created_users
    except Exception as e:
        await session.rollback()
        raise HTTPException(status_code=400, detail=str(e))