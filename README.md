# 风机耗电共用模型平台（Fan Power Model Platform）

各厂只上报工况摘要（样本数 + 整数向量和 + 外积和 + 压差零点修正量），平台用
NumPy 岭回归拟合一套共用的"转速/压差 → 每分钟耗电量"模型，按留一厂交叉验证
选 λ，SQLite 多版本保存，支持凭版号原子替换单厂修正量。

## 快速开始

```bash
docker compose up --build      # 服务监听 :8000
./demo.sh                      # 演示：建厂 → 预测 → 更正修正量 → 冲突
./demo_pilot.sh                # 演示：试点批开批/回报/更正 → 换版后旧批不变
```

本地开发（无 Docker）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn --factory app.main:create_app --port 8000
.venv/bin/python -m pytest tests/ -q     # 41 个自测
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/versions` | 上报 3–8 家厂的摘要，训练并发布新版本（201） |
| GET | `/versions/current` | 现行版：所选 λ、逐厂误差、修正量、模型 |
| GET | `/versions/{id}` | 历史版本（不可变） |
| POST | `/versions/{id}/correction` | 凭版号替换一厂修正量；成功则修正量+模型+逐厂误差作为新版本一起生效；版号冲突 409 / 重算失败 422，均保留现行版 |
| POST | `/predict` | 按厂号 + 原始转速、压差预测耗电，返回所用版号；重算期间沿用现行版。带 `batch_id` 时按该批开批时固定的版号、修正量与偏移预测 |
| POST | `/pilot/batches` | 按厂号 + 批号开试点批；重开同一批返回原固定值（201 新建 / 200 已存在） |
| GET | `/pilot/batches/{factory}/{batch}` | 查看该批固定值（版号/零点修正量/偏移）与当前摘要 |
| POST | `/pilot/batches/{factory}/{batch}/reports` | 回报（或替换）本批工况摘要，返回校准前/后均方误差 |
| GET | `/health` | 健康检查 |

### 上报格式（`POST /versions`）

```json
{
  "factories": [
    {
      "factory_id": "F1",
      "n": 1440,
      "sums":  [Σ转速, Σ压差, Σ耗电],
      "outer": [[Σs², Σsp, Σsy], [Σsp, Σp², Σpy], [Σsy, Σpy, Σy²]],
      "correction": 0
    }
  ]
}
```

`sums`/`outer` 均为**整数**（任意精度，JSON 大整数无损解析）；`correction`
为加到压差上的整数零点修正量。预测时传原始读数，平台自动加上该厂修正量。

## 试点批（检修后边试车边校准）

`POST /pilot/batches` 按厂号 + 批号为一次试车开批。开批在**同一个已提交
状态**上取数：事务一开始读现行版号指针，随后取该版号的模型、该厂的零点
修正量，并算出一个耗电偏移，三者随批永久固定。

- **偏移** = 本厂已回报摘要该版下的平均残差（实测 − 共用预测），无回报取 0。
  平均残差由 `mean(y) − (b + w'E[z])` 从**运行合计**（`report_totals`，
  每厂一行：n 与 sums）精确得出，开批不逐条扫描 `pilot_reports`。
- **重开同一批**（同厂号 + 批号）原样返回首次的版号、修正量、偏移，
  不再重算也不再写入；即使现行版已经换版。响应 201 表示新建、200 表示已存在。
- **带批号预测**：`POST /predict` 传 `batch_id` 时按该批固定的版号模型、
  零点修正量和偏移出值；共用模型换版后旧批的预测与评分依据保持不变。
- **回报**：`POST /pilot/batches/{factory}/{batch}/reports` 只收整数充分统计量
  （n/sums/outer），各批可乱序。用本批固定的版号、修正量、偏移算出
  `mse_before`（不打偏移）与 `mse_after`（打偏移），即校准前后均方误差。
- **更正**：同批再次回报即替换原摘要，重新算本批误差；同时从运行合计里
  扣除旧摘要、加入新摘要，因此新摘要参与**之后**开批的偏移计算；已开批的
  固定值与误差不受影响。校验失败（422）或事务失败时原摘要与合计原样保留。
- **换版**：共用模型发布新版本后，旧批仍按开批时的版号模型和偏移预测/评分；
  新批固定新版号 + 新零点修正量，偏移用本厂已回报摘要按**新**模型重算。

### 试点批请求示例

```bash
# 开批（F3 的 B1），未回报过 -> offset 0
curl -sS -X POST localhost:8000/pilot/batches -H 'Content-Type: application/json' \
  -d '{"factory_id":"F3","batch_id":"B1"}'
# 重开 -> 200，原固定值不变
curl -sS -X POST localhost:8000/pilot/batches -H 'Content-Type: application/json' \
  -d '{"factory_id":"F3","batch_id":"B1"}'
# 回报摘要（整数充分统计量），返回 mse_before / mse_after
curl -sS -X POST localhost:8000/pilot/batches/F3/B1/reports \
  -H 'Content-Type: application/json' --data @demo/pilot_report_b1_fixed.json
# 带批号预测
curl -sS -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"factory_id":"F3","batch_id":"B1","speed":1000064,"pressure":500001}'
```

## 关键设计

- **大基值方差保留**：协方差分子 `N·Σxxᵀ − ΣxΣxᵀ` 全程用 Python 大整数精确
  计算，最后一步才转 float64。读数基值 1e12、方差仅个位数时，朴素
  `E[x²]−E[x]²` 会被灾难性抵消成 0，这里仍精确保留（有专门测试）。
  入库前对所有厂做精确 PSD 校验（全部主子式 ≥ 0，整数判定）。
- **岭回归**：损失 = 均方误差 + λ·‖w‖²，截距免罚。特征按训练厂的 pooled
  均值与**总体标准差**标准化后，闭式解 `(C + λI)w = Cov(z,y)`，
  `b = ȳ`；零方差特征忽略。
- **留一厂验证**：轮流留一厂，只用其余厂的均值/总体标准差标准化并拟合，
  评估留出厂 MSE（由充分统计量精确求出，无需原始行）。在
  {0.01, 0.1, 1} 中选"各留出厂 MSE 最大值"最小的 λ，并列取较小者，
  再用全厂重训。
- **版本原子性**：修正量、模型、逐厂误差在同一 SQLite 事务写入，最后才翻动
  `current_version` 指针；预测只读已提交的现行版，因此重算期间预测仍沿用
  现行版。凭版号更新时若版号已过期 → 409 且现行版不变；重算失败同样保留。
- **试点批一致性**：开批事务先读现行版指针，再固定版号、修正量、偏移；
  偏移只由每厂一行的运行合计（n、sums）算出，开批是 O(1) 而不随已回报批数
  增长。回报的插入/替换、合计的扣除/加入在同一事务内完成。
- **并发**：WAL 模式 + 写锁；计算在事务外完成，提交窗口极短。

## 演示（demo.sh / demo_pilot.sh）

`scripts/make_demo_data.py` 生成确定性的 4 厂数据（转速基值 1e6、压差基值
5e5 的原始计数；F3 压差零点漂移 +500 但按 0 上报）。`demo.sh` 依次：

1. `POST /versions` 建厂 → F3 留厂误差 33.3 明显偏高，λ=0.01；
2. `POST /predict` 用原始读数预测 F3；
3. `POST /versions/1/correction` 把 F3 修正量改为 500 → 版本 2，
   逐厂误差全部降到 ≈1–2，模型与修正量同时生效；
4. 再次预测 → 结果改变且 `version=2`；
5. 用过期版号重放 → HTTP 409，现行版保持版本 2。

`demo_pilot.sh`（数据由 `scripts/make_pilot_demo_data.py` 生成到
`demo/pilot_*.json`）演示检修后的试点批：B1 开批 → 回报 → B2 开批（偏移 ≈
B1 平均残差）→ B2 乱序回报 → B1 更正摘要 → 坏请求 422 保留原摘要 → 共用模型
换版 → 旧批 B1 仍按原版预测 → 新批 B3 固定新版本与新修正量、按新模型重算
偏移 → 重开 B3 原样返回。

## 剩余事项

- 本环境无 Docker，`docker compose` 未实际构建过；Dockerfile 为标准
  python:3.12-slim 流程，建议在有 Docker 的机器上跑一次 `docker compose up --build`。
- 单进程服务（uvicorn 单 worker）；多副本部署需把 SQLite 换成集中式存储。
- 未做鉴权/审计；生产环境应加认证与操作日志。
- 留一厂验证对工况迥异的厂会外推放大误差（属预期行为）；如需可扩展分组验证。
