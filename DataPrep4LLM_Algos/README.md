## 快速开始

### git clone repo
```bash
git clone https://github.com/xxxxx.git
```

### 下载模型
[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) 布局和公式检测模型
[布局检测](https://huggingface.co/opendatalab/PDF-Extract-Kit/tree/main/models/Layout)
[公式检测](https://huggingface.co/opendatalab/PDF-Extract-Kit/tree/main/models/MFD)

[UniMERNet](https://github.com/opendatalab/UniMERNet) 公式识别模型
[公式识别](https://huggingface.co/wanderkid/unimernet_base/tree/main)

[RapidOCR](https://rapidai.github.io/RapidOCRDocs/) 文本检测和识别模型
[文本检测](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)
[文本识别](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar)

[RapidStructure](https://github.com/RapidAI/RapidStructure/blob/main/docs/README_Table.md) 表格结构预测模型
[表格结构预测](https://pan.baidu.com/share/init?surl=PI9fksW6F6kQfJhwUkewWg&pwd=p29g)

[surya](https://github.com/VikParuchuri/surya) 阅读顺序预测模型
[阅读顺序预测](https://huggingface.co/vikp/surya_order)



### 安装

> 创建conda环境

```bash
conda create -p ./ENV python=3.10
conda activate ./ENV
```

> 安装依赖包

```bash
pip install -r requirements.txt
pip install unimernet==0.2.1
pip install DataPrep4LLM_Algos/detectron2-0.6+pt2.3.1cu121-cp310-cp310-linux_x86_64.whl
``` 
或从https://miropsota.github.io/torch_packages_builder/detectron2/ 下载合适版本detectron2安装。

修改ENV/lib/python3.10/site-packages/detectron2/data/transforms/transform.py第46行"Image.LINEAR"为"IMAGE.BILINEAR"。


### 运行

```bash 
cd DataPrep4LLM_Algos
python main.py --pdf <pdf文档或文件夹>
```
如需在cpu上运行，需修改configs/layoutlmv3_base_inference.yaml中的"cuda"为"cpu"。