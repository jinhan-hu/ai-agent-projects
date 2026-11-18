"""示例知识库与评估标注集。

★ 数据是**虚构**的（一个假的运维监控产品），但结构是真实的：
    每篇文档有 doc_id / title / text / route / source，
    每条评估样本有 query / expected（期望命中的 doc_id）/ route。

为什么用虚构数据：
    1. 合规 —— 不引入任何真实客户数据
    2. 可复现 —— clone 下来跑出的指标完全一致
    3. 可标注 —— 期望命中的文档是我自己定的，评估口径清晰
"""
from __future__ import annotations

from .eval import EvalCase
from .knowledge import Doc

DOCS: list[Doc] = [
    # ═══════════ install（6 篇）═══════════
    Doc("KB-I01", "安装前置条件",
        "需要 64 位操作系统；内存至少 8 GB；磁盘剩余 20 GB。"
        "Windows 需先安装 VC++ 2015-2022 运行库；Linux 需 glibc 2.28 以上。",
        "install", "安装手册 §1"),
    Doc("KB-I02", "许可证激活步骤",
        "在「设置 → 授权 → 输入激活码」填入 32 位激活码。"
        "离线环境需先在联网机器生成请求文件，再导入许可证文件。",
        "install", "安装手册 §4"),
    Doc("KB-I03", "端口占用导致安装失败",
        "安装前确认 8080 端口空闲。若被占用，可在安装器改用 9000 端口，或先停掉占用进程。",
        "install", "安装手册 §2"),
    Doc("KB-I04", "Docker 部署方式",
        "提供官方镜像，用 docker compose up -d 启动。数据卷挂载 ./data 到 /app/data，"
        "重建容器不丢数据。",
        "install", "安装手册 §6"),
    Doc("KB-I05", "集群模式部署",
        "集群模式至少 3 个节点，需共享 Redis 做会话存储。各节点配置相同的 cluster_token。",
        "install", "安装手册 §8"),
    Doc("KB-I06", "升级与回滚",
        "升级前务必备份 data 目录。升级失败可用上一版本的安装包覆盖安装，"
        "数据目录保持不动即可回滚。",
        "install", "安装手册 §9"),

    # ═══════════ usage（6 篇）═══════════
    Doc("KB-U01", "报表模块入口",
        "报表在左侧导航「分析 → 报表中心」。「按日汇总」需选择日期区间后点右上角「生成」。",
        "usage", "用户手册 §3.2"),
    Doc("KB-U02", "告警阈值参数说明",
        "阈值支持绝对值和百分比两种。百分比模式以近 7 天均值为基准，超过阈值持续 5 分钟才触发。",
        "usage", "用户手册 §5.1"),
    Doc("KB-U03", "监控对象分组",
        "在「资产 → 分组管理」新建分组，可拖拽资产进组。分组支持嵌套，最多 3 层。",
        "usage", "用户手册 §2.4"),
    Doc("KB-U04", "通知渠道配置",
        "支持邮件、企业微信、钉钉、Webhook 四种通知渠道。"
        "Webhook 需填写 POST 地址，系统会以 JSON 格式推送告警内容。",
        "usage", "用户手册 §6.2"),
    Doc("KB-U05", "仪表盘自定义",
        "在仪表盘页点「编辑」，可拖拽添加折线图、饼图、单值卡。每个图表可绑定一个监控指标。",
        "usage", "用户手册 §4.1"),
    Doc("KB-U06", "数据保留周期设置",
        "在「设置 → 存储策略」可设置原始数据保留天数（7/30/90 天）。"
        "超过保留期的数据会被聚合后删除。",
        "usage", "用户手册 §7.3"),

    # ═══════════ troubleshoot（8 篇）═══════════
    Doc("KB-T01", "错误码 E1024",
        "E1024 表示许可证校验失败。常见原因：系统时间不准、激活码与设备指纹不匹配、许可证过期。"
        "排查顺序：先对齐系统时间，再在授权页看设备指纹是否变化。",
        "troubleshoot", "故障手册 §2"),
    Doc("KB-T02", "启动报错 ModuleNotFoundError",
        "通常是缺少依赖或虚拟环境未激活。先执行 pip install -r requirements.txt，"
        "确认 python -V 与文档要求一致。",
        "troubleshoot", "故障手册 §4"),
    Doc("KB-T03", "页面一直转圈打不开",
        "先看后端服务是否存活（curl 健康检查接口）。若超时，多为数据库连接池耗尽；"
        "可重启服务并检查慢查询日志。",
        "troubleshoot", "故障手册 §6"),
    Doc("KB-T04", "告警不触发",
        "检查三处：① 阈值是否真的被越过（看原始数据）② 是否在静默期 ③ 通知渠道是否配置正确。"
        "静默期是告警不触发最常见的原因。",
        "troubleshoot", "故障手册 §7"),
    Doc("KB-T05", "数据上报延迟",
        "先确认 Agent 端与服务端时钟偏差小于 30 秒。若 Agent 本地队列积压，"
        "多为网络抖动，可调大 batch_size 或重试间隔。",
        "troubleshoot", "故障手册 §8"),
    Doc("KB-T06", "内存占用持续上涨",
        "常见原因是长时间未清理的历史查询缓存。在「设置 → 维护」执行清理缓存；"
        "若仍上涨，检查是否有慢查询占用了大量游标。",
        "troubleshoot", "故障手册 §9"),
    Doc("KB-T07", "集群脑裂",
        "表现为多个节点都认为自己是主节点。检查 Redis 是否可用、网络是否分区。"
        "恢复后需手动指定主节点并等待从节点同步。",
        "troubleshoot", "故障手册 §10"),
    Doc("KB-T08", "登录后立即跳回登录页",
        "多为 Cookie 域配置错误或系统时间不准导致 token 校验失败。"
        "先核对 NTP 同步状态，再检查反向代理是否改写了 Cookie 的 domain。",
        "troubleshoot", "故障手册 §11"),

    # ═══════════ fallback（4 篇）═══════════
    Doc("KB-F01", "常见问题总览",
        "安装类见安装手册，使用类见用户手册，故障类见故障手册。资料未覆盖的问题建议转人工。",
        "fallback", "帮助中心首页"),
    Doc("KB-F02", "如何联系技术支持",
        "工单系统在「帮助 → 提交工单」。紧急问题可拨打 400 电话（工作日 9:00-18:00）。",
        "fallback", "帮助中心 §1"),
    Doc("KB-F03", "产品版本与许可类型",
        "分为社区版（免费，限 20 个监控对象）、专业版（按对象数订阅）、企业版（私有化部署）。",
        "fallback", "帮助中心 §2"),
    Doc("KB-F04", "账号与权限体系",
        "角色分管理员、运维、只读三种。可用 LDAP 对接企业账号，同步周期默认 1 小时。",
        "fallback", "帮助中心 §3"),
]


# ═══════════ 评估标注集 ═══════════
# 每条样本标注「期望命中哪几篇」。构造原则：
#   - 覆盖全部 4 个类别
#   - 包含「同义改写」「口语化提问」「含精确错误码」三类难度
#   - 部分样本期望命中 2 篇（考察多文档召回）

EVAL_CASES: list[EvalCase] = [
    # install（6）
    EvalCase("装之前要准备什么环境", {"KB-I01"}, "install"),
    EvalCase("激活码在哪里填", {"KB-I02"}, "install"),
    EvalCase("8080 被别的程序占了怎么办", {"KB-I03"}, "install"),
    EvalCase("能用 docker 起吗", {"KB-I04"}, "install"),
    EvalCase("几台机器能做集群", {"KB-I05"}, "install"),
    EvalCase("升级失败怎么退回去", {"KB-I06"}, "install"),
    # usage（6）
    EvalCase("报表在哪个菜单下面", {"KB-U01"}, "usage"),
    EvalCase("阈值百分比模式是什么意思", {"KB-U02"}, "usage"),
    EvalCase("怎么把资产分组", {"KB-U03"}, "usage"),
    EvalCase("能发企业微信通知吗", {"KB-U04"}, "usage"),
    EvalCase("仪表盘能自己加图表吗", {"KB-U05"}, "usage"),
    EvalCase("数据存多久，能调吗", {"KB-U06"}, "usage"),
    # troubleshoot（8）
    EvalCase("启动报 E1024", {"KB-T01"}, "troubleshoot"),
    EvalCase("提示 ModuleNotFoundError", {"KB-T02"}, "troubleshoot"),
    EvalCase("页面一直转圈", {"KB-T03"}, "troubleshoot"),
    EvalCase("配了告警但一直没收到通知", {"KB-T04"}, "troubleshoot"),
    EvalCase("数据上报有延迟", {"KB-T05"}, "troubleshoot"),
    EvalCase("内存一直往上涨", {"KB-T06"}, "troubleshoot"),
    EvalCase("集群出现两个主节点", {"KB-T07"}, "troubleshoot"),
    EvalCase("登录后马上又跳回登录页", {"KB-T08"}, "troubleshoot"),
    # fallback（4）
    EvalCase("有没有免费版", {"KB-F03"}, "fallback"),
    EvalCase("怎么提交工单找人帮忙", {"KB-F02"}, "fallback"),
    EvalCase("账号角色有哪几种", {"KB-F04"}, "fallback"),
    EvalCase("文档看完了还是不会", {"KB-F01"}, "fallback"),
]
