# auth/two_factor.py
"""
Two-Factor Authentication (2FA) module for Eagle Framework.
"""
import pyotp
import qrcode
import io
import base64
from typing import Optional, List
from pydantic import BaseModel
from sqlalchemy import String, Boolean, Text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status, Depends, APIRouter
from ..db import BaseModel as DBBaseModel, get_db, Mapped, mapped_column
from ..admin import register_model_to_admin
from . import get_current_active_user, AuthUser


class TwoFactorSetup(BaseModel):
    """2FA setup response."""
    secret: str
    qr_code: str
    backup_codes: List[str]

class TwoFactorVerify(BaseModel):
    """2FA verification request."""
    code: str

class TwoFactorDisable(BaseModel):
    """2FA disable request."""
    password: str

@register_model_to_admin
class TwoFactorAuth(DBBaseModel):
    """Two-Factor Authentication model."""
    __tablename__ = "two_factor_auth"
    
    user_id: Mapped[int] = mapped_column(unique=True, nullable=False)
    secret_key: Mapped[str] = mapped_column(String(32), nullable=False)
    backup_codes: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    last_used_code: Mapped[Optional[str]] = mapped_column(String(6), nullable=True)

class TwoFactorService:
    """Two-Factor Authentication service."""
    
    @staticmethod
    def generate_secret() -> str:
        """Generate a new TOTP secret."""
        return pyotp.random_base32()
    
    @staticmethod
    def generate_backup_codes(count: int = 10) -> List[str]:
        """Generate backup codes."""
        return [secrets.token_hex(4).upper() for _ in range(count)]
    
    @staticmethod
    def generate_qr_code(secret: str, user_email: str, issuer: str = "Eagle Framework") -> str:
        """Generate QR code for TOTP setup."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    @staticmethod
    def verify_totp(secret: str, code: str, last_used: Optional[str] = None) -> bool:
        """Verify TOTP code."""
        if code == last_used:
            return False  # Prevent code reuse
        
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)
    
    @staticmethod
    def verify_backup_code(backup_codes: List[str], code: str) -> bool:
        """Verify backup code."""
        return code.upper() in [c.upper() for c in backup_codes]

async def get_user_2fa(db: AsyncSession, user_id: int) -> Optional[TwoFactorAuth]:
    """Get user's 2FA settings."""
    result = await db.execute(
        select(TwoFactorAuth).where(TwoFactorAuth.user_id == user_id)
    )
    return result.scalar_one_or_none()

# 2FA Router
two_factor_router = APIRouter(prefix="/auth/2fa", tags=["two-factor-auth"])

@two_factor_router.post("/setup", response_model=TwoFactorSetup)
async def setup_2fa(
    current_user: AuthUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Setup 2FA for user."""
    # Check if 2FA is already enabled
    existing_2fa = await get_user_2fa(db, current_user.id)
    if existing_2fa and existing_2fa.is_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled"
        )
    
    service = TwoFactorService()
    secret = service.generate_secret()
    backup_codes = service.generate_backup_codes()
    qr_code = service.generate_qr_code(secret, current_user.email)
    
    # Save 2FA settings (disabled by default until verified)
    if existing_2fa:
        existing_2fa.secret_key = secret
        existing_2fa.backup_codes = json.dumps(backup_codes)
        existing_2fa.is_enabled = False
    else:
        two_fa = TwoFactorAuth(
            user_id=current_user.id,
            secret_key=secret,
            backup_codes=json.dumps(backup_codes),
            is_enabled=False
        )
        db.add(two_fa)
    
    await db.commit()
    
    return TwoFactorSetup(
        secret=secret,
        qr_code=qr_code,
        backup_codes=backup_codes
    )

@two_factor_router.post("/enable")
async def enable_2fa(
    verify_data: TwoFactorVerify,
    current_user: AuthUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Enable 2FA after verification."""
    two_fa = await get_user_2fa(db, current_user.id)
    if not two_fa:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA not set up"
        )
    
    service = TwoFactorService()
    if not service.verify_totp(two_fa.secret_key, verify_data.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid 2FA code"
        )
    
    two_fa.is_enabled = True
    two_fa.last_used_code = verify_data.code
    await db.commit()
    
    return {"message": "2FA enabled successfully"}

@two_factor_router.post("/verify")
async def verify_2fa(
    verify_data: TwoFactorVerify,
    current_user: AuthUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Verify 2FA code."""
    two_fa = await get_user_2fa(db, current_user.id)
    if not two_fa or not two_fa.is_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA not enabled"
        )
    
    service = TwoFactorService()
    
    # Try TOTP first
    if service.verify_totp(two_fa.secret_key, verify_data.code, two_fa.last_used_code):
        two_fa.last_used_code = verify_data.code
        await db.commit()
        return {"message": "2FA verification successful", "method": "totp"}
    
    # Try backup code
    backup_codes = json.loads(two_fa.backup_codes)
    if service.verify_backup_code(backup_codes, verify_data.code):
        # Remove used backup code
        backup_codes.remove(verify_data.code.upper())
        two_fa.backup_codes = json.dumps(backup_codes)
        await db.commit()
        return {"message": "2FA verification successful", "method": "backup"}
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid 2FA code"
    )

@two_factor_router.post("/disable")
async def disable_2fa(
    disable_data: TwoFactorDisable,
    current_user: AuthUser = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Disable 2FA."""
    from . import verify_password
    
    if not verify_password(disable_data.password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid password"
        )
    
    two_fa = await get_user_2fa(db, current_user.id)
    if two_fa:
        await db.delete(two_fa)
        await db.commit()
    
    return {"message": "2FA disabled successfully"}
