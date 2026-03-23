"""EDA 报告代理接口。"""

from urllib import error as urlerror
from urllib import request as urlrequest

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

router = APIRouter(prefix="/api/eda", tags=["eda"])


@router.get("/report/{thread_id}")
async def get_eda_report(request: Request, thread_id: str, download: bool = Query(False)):
    """代理获取 EDA 后端 HTML 报告，并支持下载。"""

    base_url = request.app.state.config.eda_api_base_url.rstrip("/")
    endpoint = f"{base_url}/eda-report/{thread_id}"

    try:
        req = urlrequest.Request(endpoint, method="GET")
        with urlrequest.urlopen(req, timeout=request.app.state.config.request_timeout) as resp:
            html_bytes = resp.read()
    except urlerror.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=f"EDA 报告获取失败: HTTP {exc.code}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"EDA 报告获取失败: {str(exc)}") from exc

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="eda_report_{thread_id}.html"'

    return Response(content=html_bytes, media_type="text/html; charset=utf-8", headers=headers)


@router.get("/download-csv/{thread_id}")
async def download_eda_csv(request: Request, thread_id: str):
    """代理下载 EDA 后端当前清洗后的 CSV。"""

    base_url = request.app.state.config.eda_api_base_url.rstrip("/")
    endpoint = f"{base_url}/download-csv/{thread_id}"

    try:
        req = urlrequest.Request(endpoint, method="GET")
        with urlrequest.urlopen(req, timeout=request.app.state.config.request_timeout) as resp:
            csv_bytes = resp.read()
    except urlerror.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=f"CSV 下载失败: HTTP {exc.code}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CSV 下载失败: {str(exc)}") from exc

    headers = {
        "Content-Disposition": f'attachment; filename="cleaned_{thread_id}.csv"',
    }
    return Response(content=csv_bytes, media_type="text/csv; charset=utf-8", headers=headers)
