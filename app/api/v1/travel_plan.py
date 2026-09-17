from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.plan_service import PlanService

router = APIRouter()


@router.get("/travel-plans")
async def list_plans(user_id: str = "anonymous", db: AsyncSession = Depends(get_db)):
    """获取用户方案列表"""
    return await PlanService.list_plans(user_id, db)


@router.get("/travel-plans/{plan_id}")
async def get_plan(plan_id: str, db: AsyncSession = Depends(get_db)):
    """获取方案详情"""
    return await PlanService.get_plan(plan_id, db)


@router.post("/travel-plans")
async def create_plan(plan_data: dict, db: AsyncSession = Depends(get_db)):
    """创建方案"""
    return await PlanService.create_plan(plan_data, db)


@router.put("/travel-plans/{plan_id}")
async def update_plan(plan_id: str, plan_data: dict, db: AsyncSession = Depends(get_db)):
    """更新方案"""
    return await PlanService.update_plan(plan_id, plan_data, db)


@router.get("/plans/{plan_id}/file")
async def get_plan_file(plan_id: str, db: AsyncSession = Depends(get_db)):
    """获取方案 Markdown 文件"""
    return await PlanService.get_plan_file(plan_id, db)


@router.get("/plans/{plan_id}/preview")
async def preview_plan(plan_id: str, db: AsyncSession = Depends(get_db)):
    """预览方案 HTML"""
    return await PlanService.preview_plan(plan_id, db)


@router.get("/plans/{plan_id}/versions")
async def get_versions(plan_id: str, db: AsyncSession = Depends(get_db)):
    """版本历史"""
    return await PlanService.get_versions(plan_id, db)


@router.delete("/travel-plans/{plan_id}")
async def delete_plan(plan_id: str, db: AsyncSession = Depends(get_db)):
    """删除方案及其文件"""
    return await PlanService.delete_plan(plan_id, db)


@router.delete("/plans/{plan_id}/file")
async def delete_plan_file(plan_id: str, version: int = None, db: AsyncSession = Depends(get_db)):
    """删除方案文件"""
    return await PlanService.delete_plan_file(plan_id, version, db)