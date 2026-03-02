# Scripts / 工具脚本

本目录包含项目维护和文档同步工具。

## update_docs.py - 文档自动更新工具

当代码更改或有新的训练/实验时，运行此脚本同步更新 README 文档。

### 使用方法

```bash
# 更新所有文档
python scripts/update_docs.py

# 仅更新特定模块
python scripts/update_docs.py --core          # core/README.md
python scripts/update_docs.py --training      # training/README.md
python scripts/update_docs.py --experiments   # docs/experiments.md

# 生成更新日志
python scripts/update_docs.py --changelog
```

### 自动同步的内容

| 模块 | 同步内容 | 来源 |
|------|----------|------|
| core/README.md | 模型参数默认值 | core/pim_ik_net.py |
| training/README.md | 训练参数默认值 | training/trainer.py |
| docs/experiments.md | 实验结果数据 | 实验运行结果 |
| 主 README | 更新日志 | Git commit 历史 |

### 工作流程建议

1. **修改代码后**
   ```bash
   # 1. 修改代码 (如修改模型参数)
   vim core/pim_ik_net.py

   # 2. 运行文档更新
   python scripts/update_docs.py --core

   # 3. 提交更改
   git add core/pim_ik_net.py core/README.md
   git commit -m "update: 模型参数调整"
   ```

2. **训练新模型后**
   ```bash
   # 1. 训练完成，获得新结果

   # 2. 更新实验文档
   python scripts/update_docs.py --experiments

   # 3. 提交
   git add docs/experiments.md
   git commit -m "experiments: 添加新的训练结果"
   ```

3. **发布新版本前**
   ```bash
   # 运行完整更新
   python scripts/update_docs.py

   # 检查所有文档
   git status
   ```

### 扩展

要添加新的文档同步规则，编辑 `update_docs.py` 并添加新的 `update_xxx_readme()` 函数。
