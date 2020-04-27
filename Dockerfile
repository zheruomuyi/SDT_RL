# 基于远景智能基础镜像构建：java7_centos7.4
FROM harbor.eniot.io/arch/tensorflow-cpu-py3:tag_tensorflow-cpu-py3_20181224_001

# MAINTAINER
MAINTAINER jinjin.liu <jinjin.liu@envision-digital.com>

# add dependence
RUN pip install -r /code/requirements.txt -i

# 定义环境变量
ENV APP_NAME tsdb-compress-rl
ENV APP_HOME /home/envuser/$APP_NAME
RUN mkdir -p $APP_HOME/bin $APP_HOME/logs $APP_HOME/conf
RUN mkdir -p /data/apps/$APP_NAME

# 复制文件夹
COPY main $APP_HOME/bin/
COPY ./startup.sh $APP_HOME/

# 指定工作目录
WORKDIR $APP_HOME

# 执行命令，改变文件权限
RUN chmod 755 startup.sh

# 指定容器启动程序及参数
ENTRYPOINT ["bash","./startup.sh"]
