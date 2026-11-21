"""知识库生成器：在 24 篇手写核心文档之上，生成 476 篇长尾文档。

为什么这样设计：
    真实企业知识库的结构就是「少量核心文档 + 大量长尾文档」。
    只用手写的 24 篇，评估太容易（每条 query 都能找到）；
    只用生成的文档，语义太模板化，评估又太假。
    ★ 所以两者混合，才接近真实分布。

生成规则（确定性，无随机，保证可复现）：
    10 个产品模块 × 4 类问题 × 12 个细分场景 = 480 篇，
    取前 476 篇，加上手写 24 篇 = 500 篇。
"""
from __future__ import annotations

from .eval import EvalCase
from .knowledge import Doc
from .knowledge_data import DOCS as CORE_DOCS
from .knowledge_data import EVAL_CASES as CORE_CASES

MODULES = [
    ("账号", "account", ["登录", "密码", "SSO", "LDAP", "二次验证", "会话", "锁定", "注销", "注册", "重置", "绑定", "过户"]),
    ("报表", "report", ["汇总", "明细", "导出", "定时", "订阅", "对比", "环比", "同比", "透视", "钻取", "模板", "缓存"]),
    ("告警", "alert", ["阈值", "静默", "升级", "收敛", "分组", "去重", "抑制", "恢复", "风暴", "依赖", "路由", "订阅"]),
    ("资产", "asset", ["录入", "导入", "标签", "分组", "拓扑", "依赖", "变更", "巡检", "下线", "回收", "同步", "合并"]),
    ("通知", "notify", ["邮件", "短信", "企微", "钉钉", "Webhook", "回执", "重试", "限频", "模板", "签名", "队列", "失败"]),
    ("存储", "storage", ["保留期", "归档", "压缩", "冷热", "扩容", "迁移", "副本", "快照", "清理", "配额", "对象", "索引"]),
    ("集群", "cluster", ["部署", "扩容", "缩容", "主从", "脑裂", "选主", "心跳", "分区", "升级", "回滚", "节点", "令牌"]),
    ("权限", "rbac", ["角色", "继承", "最小权限", "越权", "审批", "临时", "委托", "审计", "跨租户", "标签", "策略", "回收"]),
    ("审计", "audit", ["操作日志", "登录日志", "导出", "留存", "检索", "告警", "篡改", "归档", "报表", "合规", "追溯", "脱敏"]),
    ("集成", "integration", ["API", "Webhook", "SDK", "OAuth", "回调", "幂等", "限流", "签名", "重放", "灰度", "沙箱", "对账"]),
]

# 每类 route 的「问题描述模板」和「解决步骤模板」
ROUTE_TEMPLATES = {
    "install": {
        "title": "{module}模块·{scene}的部署与启用",
        "body": (
            "部署{module}模块的{scene}能力前，需确认：64 位系统、内存 ≥ 8 GB、"
            "磁盘剩余 20 GB。启用步骤：① 在「设置 → 模块」勾选{module}；"
            "② 配置{scene}相关参数（默认值可直接使用）；③ 保存后重启服务生效。"
            "常见问题：模块未出现在列表中，通常是许可证未包含该模块。"
        ),
    },
    "usage": {
        "title": "{module}模块·{scene}怎么用",
        "body": (
            "{module}的{scene}入口在左侧导航「{module} → {scene}」。"
            "操作路径：选择对象 → 设置{scene}参数 → 点「保存」。"
            "参数说明：{scene}默认按近 7 天均值计算；"
            "可切换为绝对值模式。修改后约 1 分钟生效。"
        ),
    },
    "troubleshoot": {
        "title": "{module}模块·{scene}异常排查",
        "body": (
            "{module}的{scene}出现异常时，按以下顺序排查："
            "① 确认服务存活（健康检查接口）；② 查看该模块的错误日志；"
            "③ 核对{scene}相关配置是否被改动。"
            "绝大多数{scene}异常源于配置漂移或资源不足（连接池耗尽、磁盘写满）。"
        ),
    },
    "fallback": {
        "title": "{module}模块·{scene}常见疑问",
        "body": (
            "关于{module}模块的{scene}：相关内容分布在安装手册、用户手册与故障手册中。"
            "若资料未覆盖你的情形，建议在「帮助 → 提交工单」提交，"
            "附上模块名称、{scene}的具体现象与时间点，便于定位。"
        ),
    },
}

# 生成时给每篇文档一个可辨识的「关键词锚」，使评估有区分度
ANCHORS = ["基础版", "专业版", "企业版", "私有化", "公有云", "混合部署",
           "V4 版本", "旧版本", "新版本", "灰度环境", "生产环境", "测试环境"]

ERROR_CODES = ["E2001", "E2002", "E2003", "E2004", "E2005", "E2006",
               "E2007", "E2008", "E2009", "E2010", "E2011", "E2012"]


def _build() -> tuple[list[Doc], list[EvalCase]]:
    docs: list[Doc] = []
    cases: list[EvalCase] = []
    n = 0

    for route, tpl in ROUTE_TEMPLATES.items():
        for mname, mkey, scenes in MODULES:
            for i, scene in enumerate(scenes):
                if len(docs) >= 476:
                    break
                n += 1
                anchor = ANCHORS[i % len(ANCHORS)]
                code = ERROR_CODES[i % len(ERROR_CODES)]
                doc_id = f"KB-{route[:2].upper()}{mkey.upper()[:4]}-{n:03d}"
                title = tpl["title"].format(module=mname, scene=scene) + f"（{anchor}）"
                body = tpl["body"].format(module=mname, scene=scene)
                # 注入差异化信息，让每篇文档有独特可检索特征
                body += f" 适用环境：{anchor}。相关错误码：{code}。"
                if route == "troubleshoot":
                    body += f" 排查{scene}时优先检查 {code} 相关指标。"
                docs.append(
                    Doc(doc_id, title, body, route, f"{mname}手册 §{i + 1}.{n % 9 + 1}")
                )
                # 每条文档配一条 query：用「模块 + 场景 + 锚点」组合，模拟真实提问
                q = f"{mname}{scene}{anchor}怎么处理"
                cases.append(EvalCase(q, {doc_id}, route))
    return docs, cases


GEN_DOCS, GEN_CASES = _build()

# 最终知识库与评估集
ALL_DOCS: list[Doc] = CORE_DOCS + GEN_DOCS
ALL_CASES: list[EvalCase] = CORE_CASES + GEN_CASES
