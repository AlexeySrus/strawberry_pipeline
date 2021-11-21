# Реализация задачи Digital strawberry от команды RostovNats

Данный сервис запущен для использования по следующему адресу: 
[http://95.213.247.247:8080/](http://95.213.247.247:8501/ "RostovNats")

## Инструкция по запуску докерезированного решения
#### Вы можете скачать готовый образ докера
Для это загрузите его по следующей ссылке: [https://disk.yandex.ru/d/IOSxJ1SHOQxU5g](https://disk.yandex.ru/d/IOSxJ1SHOQxU5g "Yandex Disk")

Затем необходимо загрузить данный файл образа:
```shell
sudo docker load --input rostovnats_strawberry_pipeline.tar
```

#### Или же собрать его самостоятельно
1. Необходимо склонировать данный репозиторий с помощью следующей команды:
```shell
git clone https://github.com/AlexeySrus/strawberry_pipeline.git
```
2. Далее необходимо запустить команду по сборке образа:
```shell
sudo docker build -t rostovnats_strawberry_pipeline .
```


#### Запуск приложения
Для запуска приложения запустите следующую команду:
```shell
sudo docker run -p 8000:8000 -p 8080:8080 -d rostovnats_strawberry_pipeline
```

После этого откройте в вашем браузере следующую страницу: 
[http://localhost:8080/](http://localhost:8080/ "RostovNats")