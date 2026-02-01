# Local Wikipedia Setup Guide

本指南介绍如何设置本地Wikipedia知识库，用于user simulator的长度控制。

## 快速开始

### 方法1: 自动化脚本（推荐）

```bash
# 1. 下载、提取并构建索引（一键完成）
cd src/simulator
python prepare_wikipedia.py --lang en --output ../../data/wikipedia

# 这个过程包括：
# - 下载Wikipedia转储文件（~20GB，需要1-2小时）
# - 提取文章到JSON格式（需要1-2小时）
# - 构建索引（需要30分钟）
```

### 方法2: 手动步骤

如果自动脚本失败，可以手动执行：

#### 步骤1: 下载Wikipedia转储

```bash
# 创建目录
mkdir -p data/wikipedia

# 下载英文Wikipedia（~20GB压缩）
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
    -O data/wikipedia/enwiki-latest-pages-articles.xml.bz2

# 或者下载中文Wikipedia（~3GB压缩）
wget https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2 \
    -O data/wikipedia/zhwiki-latest-pages-articles.xml.bz2
```

#### 步骤2: 安装WikiExtractor

```bash
pip install wikiextractor
```

#### 步骤3: 提取文章

```bash
# 提取到JSON格式
python -m wikiextractor.WikiExtractor \
    --json \
    --no-templates \
    --processes 4 \
    --output data/wikipedia/extracted \
    data/wikipedia/enwiki-latest-pages-articles.xml.bz2
```

#### 步骤4: 构建索引

```python
from src.simulator.local_wikipedia import LocalWikipedia

# 首次运行会自动构建索引
wiki = LocalWikipedia(
    wiki_dir="data/wikipedia/extracted",
    rebuild_index=True
)

# 索引会保存到 data/wikipedia/extracted/wiki_index.pkl
```

## 使用方法

### 基本使用

```python
from src.simulator.local_wikipedia import LocalWikipedia

# 初始化（首次会构建索引，后续会加载缓存）
wiki = LocalWikipedia(wiki_dir="data/wikipedia/extracted")

# 获取摘要
summary = wiki.get_summary("elephant", sentences=2)
print(summary)
# 输出: "An elephant is a large mammal of the family Elephantidae.
#        They are the largest existing land animals."

# 搜索相关词条
results = wiki.search("machine learning", top_k=5)
for title, summary in results:
    print(f"{title}: {summary[:100]}...")
```

### 集成到AdaptiveLengthController

```python
from src.simulator.adaptive_length_controller import AdaptiveLengthController

# 使用本地Wikipedia
controller = AdaptiveLengthController(
    target_range=(50, 300),
    num_bins=5,
    use_local_wiki=True,
    wiki_dir="data/wikipedia/extracted"
)

# 使用
message = "What do you see in the image?"
target_length = controller.get_target_length()
padded = controller.pad_to_target(message, target_length)
```

### 性能优化

```python
# 使用更大的缓存
wiki = LocalWikipedia(
    wiki_dir="data/wikipedia/extracted",
    cache_size=5000  # 默认1000
)

# 查看统计信息
stats = wiki.get_stats()
print(f"Total articles: {stats['total_articles']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Index size: {stats['index_file_size_mb']:.2f} MB")

# 清空缓存（如果内存不足）
wiki.clear_cache()
```

## 文件结构

```
data/wikipedia/
├── enwiki-latest-pages-articles.xml.bz2  # 原始转储（~20GB）
└── extracted/
    ├── AA/
    │   ├── wiki_00
    │   ├── wiki_01
    │   └── ...
    ├── AB/
    │   └── ...
    └── wiki_index.pkl  # 索引文件（~5GB）
```

## 索引格式

索引是一个Python字典，格式如下：

```python
{
    "elephant": {
        "title": "Elephant",  # 原始大小写
        "summary": "An elephant is a large mammal..."
    },
    "machine learning": {
        "title": "Machine learning",
        "summary": "Machine learning is a field of study..."
    },
    ...
}
```

## 常见问题

### Q1: 下载太慢怎么办？

可以使用镜像站点或者使用断点续传：

```bash
# 使用aria2c（支持多线程下载）
aria2c -x 16 -s 16 \
    https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

### Q2: 磁盘空间不够怎么办？

- 压缩文件：~20GB
- 提取后：~60GB
- 索引文件：~5GB
- **总共需要：~85GB**

如果空间不足，可以：
1. 删除原始压缩文件（提取后）
2. 使用更小的语言版本（如中文~3GB）
3. 只提取部分文章

### Q3: 提取太慢怎么办？

```bash
# 增加进程数（根据CPU核心数）
python -m wikiextractor.WikiExtractor \
    --processes 8 \
    ...
```

### Q4: 如何更新Wikipedia数据？

```bash
# 重新下载最新转储
python prepare_wikipedia.py --lang en --output data/wikipedia

# 或者手动下载后重建索引
python -c "from local_wikipedia import LocalWikipedia; \
           LocalWikipedia('data/wikipedia/extracted', rebuild_index=True)"
```

### Q5: 如何只使用常用词条（节省空间）？

创建精简版索引：

```python
from src.simulator.local_wikipedia import LocalWikipedia
import pickle

# 加载完整索引
wiki = LocalWikipedia("data/wikipedia/extracted")

# 只保留常用词条（例如：长度<100字符的标题）
common_terms = {
    k: v for k, v in wiki.index.items()
    if len(k) < 100
}

# 保存精简索引
with open("data/wikipedia/extracted/wiki_index_lite.pkl", 'wb') as f:
    pickle.dump(common_terms, f)

print(f"Reduced from {len(wiki.index)} to {len(common_terms)} terms")
```

## 性能基准

在标准硬件上（Intel i7, 16GB RAM, SSD）：

- **索引构建时间**: ~30分钟（600万文章）
- **索引加载时间**: ~5秒
- **查询速度**:
  - 精确匹配: <1ms
  - 模糊匹配: <10ms
  - 搜索: <100ms
- **内存占用**:
  - 索引: ~3GB
  - 缓存(1000条): ~10MB

## 替代方案

如果不想下载完整Wikipedia，可以使用在线API：

```python
# 使用wikipedia-api库
pip install wikipedia-api

from wikipedia import Wikipedia

wiki = Wikipedia('en')
page = wiki.page('Elephant')
print(page.summary)
```

但注意：
- ❌ 需要网络连接
- ❌ 有API速率限制
- ❌ 查询较慢（~500ms）
- ✅ 不占用磁盘空间
- ✅ 始终是最新数据

## 下一步

设置完成后，可以：

1. 测试Wikipedia查询：
   ```bash
   python -c "from src.simulator.local_wikipedia import get_wikipedia; \
              wiki = get_wikipedia(); \
              print(wiki.get_summary('python'))"
   ```

2. 集成到simulator：
   参考 [adaptive_length_controller.py](adaptive_length_controller.py)

3. 运行benchmark：
   ```bash
   python run_benchmark.py --enable_length_control
   ```
