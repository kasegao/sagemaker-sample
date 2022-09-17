# sagemaker-sample

SageMaker Notebook で PyTorch を使用して機械学習を行うときの自分用サンプル。

## 概要

`src/` 以下に通常の機械学習タスクで使用するようなプログラムを設置している。これについては、一部環境変数などを除いてそこまで SageMaker 専用にプログラムを作成する必要はない。

`main.ipynb` に SageMaker 向けのプログラムを実装している。ここでは S3 と連携したデータのやり取りや、TrainingJob の作成などを行う。

## 追加のライブラリ

プログラム内で SageMaker にデフォルトで梱包されていないライブラリを使用する場合は、 `source_dir` 内に `requirements.txt` を設置すれば、 コンテナ立ち上げ時に自動でインストールされる。

`main.ipynb`

```python
estimator = PyTorch(
    entry_point="entry.py",
    source_dir="src",
    # ...
)
```

`src/requirements.txt`

```txt
dataclasses-json==0.5.7
```

## 参考

### 環境変数

色々あるので公式資料参照  
<https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md>

### GitHub 連携

<https://aws.amazon.com/jp/blogs/news/amazon-sagemaker-notebooks-now-support-git-integration-for-increased-persistence-collaboration-and-reproducibility/>

### ライフサイクル設定

- [1 時間 idle 状態だと自動でインスタンスを停止させるスクリプト](./sagemaker/auto-stop-idle.sh)
- [インスタンス起動時に追加のライブラリを install するスクリプト](./sagemaker/auto-stop-idle.sh)

※ Windows 環境の人は改行コードに気を付けてコピペしないと `/bin/bash^M: bad interpreter: No such file or directory` で怒られるので注意。  
参考: <https://github.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/issues/8>
