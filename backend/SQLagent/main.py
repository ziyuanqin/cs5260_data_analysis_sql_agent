
import os
from sqlalchemy import create_engine
from backend.SQLagent.data_expert_SQL import create_smart_sql_graph
from sqlalchemy.pool import StaticPool

def get_sql_graph_app(db_type="sqlite", mysql_url=None):
    """
    动态工厂函数：根据需求创建不同的数据库引擎和 Graph 实例
    """
    if db_type == "sqlite":
        # 内存模式，适合处理前端临时上传的 CSV
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
    elif db_type == "mysql" and mysql_url:
        # 生产模式，连接固定数据库
        engine = create_engine(
            mysql_url,
            pool_recycle=3600,
            pool_pre_ping=True
        )
    else:
        raise ValueError("不支持的数据库类型或缺少连接配置")

    # 返回针对该 engine 编译好的编译 App
    return create_smart_sql_graph(engine)