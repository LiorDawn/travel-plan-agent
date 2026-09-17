"""行程方案用例编排 — 封装方案 CRUD 与 Markdown 文件，供 API 层薄壳调用"""
from typing import List
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import TravelPlan
from app.memory import long_term
from app.files.generator import generate_markdown, markdown_to_html
from app.core.errors import NotFoundError


class PlanService:
    """行程方案管理用例"""

    @staticmethod
    async def list_plans(user_id: str, db: AsyncSession) -> List[dict]:
        plans = await long_term.get_user_plans(db, user_id)
        return [{
            "id": p.id, "title": p.title, "destination": p.destination,
            "status": p.status, "created_at": str(p.created_at),
        } for p in plans]

    @staticmethod
    async def get_plan(plan_id: str, db: AsyncSession) -> dict:
        plan = (await db.execute(select(TravelPlan).where(TravelPlan.id == plan_id))).scalar_one_or_none()
        if not plan:
            raise NotFoundError("方案不存在")
        return {"id": plan.id, "title": plan.title, "plan_data": plan.plan_data, "status": plan.status}

    @staticmethod
    async def create_plan(plan_data: dict, db: AsyncSession) -> dict:
        plan = await long_term.save_travel_plan(db, plan_data)
        return {"id": plan.id, "status": "created"}

    @staticmethod
    async def update_plan(plan_id: str, plan_data: dict, db: AsyncSession) -> dict:
        await db.execute(
            update(TravelPlan)
            .where(TravelPlan.id == plan_id)
            .values(plan_data=plan_data, title=plan_data.get("title", ""))
        )
        await db.commit()
        return {"id": plan_id, "status": "updated"}

    @staticmethod
    async def get_plan_file(plan_id: str, db: AsyncSession) -> dict:
        files = await long_term.get_plan_files(db, plan_id)
        if not files:
            plan = (await db.execute(select(TravelPlan).where(TravelPlan.id == plan_id))).scalar_one_or_none()
            if not plan:
                raise NotFoundError("方案不存在")
            md = generate_markdown(plan.plan_data)
            await long_term.save_plan_file(db, plan_id, md)
            return {"plan_id": plan_id, "version": 1, "content": md}
        return {"plan_id": plan_id, "version": files[0].version, "content": files[0].content}

    @staticmethod
    async def preview_plan(plan_id: str, db: AsyncSession) -> dict:
        files = await long_term.get_plan_files(db, plan_id)
        if not files:
            raise NotFoundError("方案文件不存在")
        return {"plan_id": plan_id, "html": markdown_to_html(files[0].content)}

    @staticmethod
    async def get_versions(plan_id: str, db: AsyncSession) -> List[dict]:
        files = await long_term.get_plan_files(db, plan_id)
        return [{
            "version": f.version, "hash": f.content_hash,
            "summary": f.change_summary, "created_at": str(f.created_at),
        } for f in files]

    @staticmethod
    async def delete_plan(plan_id: str, db: AsyncSession) -> dict:
        ok = await long_term.delete_plan(db, plan_id)
        if not ok:
            raise NotFoundError("方案不存在")
        return {"id": plan_id, "status": "deleted"}

    @staticmethod
    async def delete_plan_file(plan_id: str, version: int = None, db: AsyncSession = None) -> dict:
        ok = await long_term.delete_plan_file(db, plan_id, version)
        if not ok:
            raise NotFoundError("文件不存在")
        return {"plan_id": plan_id, "status": "deleted"}