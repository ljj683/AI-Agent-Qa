# AI-Agent-Qa 常见报错与解决方法

## 1. No module named 'src'

原因：没有从项目根目录运行，或 Python 路径没有包含项目根目录。

解决：进入项目根目录后运行：

```bash
streamlit run src/streamlit_app.py
```

不要进入 `src` 目录里面运行。

## 2. StreamlitSecretNotFoundError

原因：本地没有 `.streamlit/secrets.toml`。

解决：本项目代码已经做了 try/except 兼容，本地主要使用 `.env`。确保项目根目录有 `.env`。

## 3. DeepSeek API Key 未配置

现象：页面显示“当前未配置 DeepSeek API Key”。

解决：本地检查 `.env`：

```env
DEEPSEEK_API_KEY=你的真实Key
```

Streamlit Cloud 检查 Secrets：

```toml
DEEPSEEK_API_KEY = "你的真实Key"
```

## 4. 模型名错误

现象：DeepSeek 报 model not found / invalid model。

推荐模型名：

```text
deepseek-v4-flash
```

如果不可用，改成：

```text
deepseek-chat
```

不要写：

```text
DeepSeek-V4pro
DeepSeek-V4-Flash
v4flash
deepseek_v4_flash
```

## 5. PDF 解析文本过少

原因：PDF 是扫描版或图片型 PDF。

解决：

1. 先用 MinerU / OCR 转 Markdown 或 TXT
2. 再上传 `.md` 或 `.txt`

## 6. Word 解析失败

原因：文件不是标准 `.docx`，或者是旧版 `.doc`。

解决：

1. 用 Word/WPS 打开
2. 另存为 `.docx`
3. 重新上传

## 7. 上传后问题回答很空泛

可能原因：

1. 文档解析文本太少
2. 用户问题和文档内容无关
3. DeepSeek Key 未配置，系统返回降级片段

解决：

1. 检查解析字符数和文本块数量
2. 换一个文档中明确存在的问题
3. 检查 Key 和模型名

## 8. Streamlit Cloud 部署失败

常见原因：

1. `requirements.txt` 拼写错误
2. GitHub 没有上传 `storage/`
3. GitHub 没有上传 `data/parsed/`
4. Secrets 没填
5. Python 版本过低

解决：

- 确保 `requirements.txt` 包含：

```text
PyMuPDF
python-docx
openai
streamlit
scikit-learn
joblib
```

- 确保 `storage/` 在 GitHub 中存在。

## 9. sklearn/joblib 加载模型失败

原因：模型文件缺失或版本差异较大。

解决：重新运行：

```bash
python -m src.train_text_classifier
python -m src.text_preprocess
python -m src.build_vector_store
```

## 10. Git push 失败

如果提示网络连接 GitHub 失败：

1. 换网络或手机热点
2. 检查代理
3. 稍后重试：

```bash
git push
```

## 11. .env 被误传 GitHub

解决：立即删除远程仓库中的 `.env`，重新生成 DeepSeek Key，并确认 `.gitignore` 有：

```gitignore
.env
```
