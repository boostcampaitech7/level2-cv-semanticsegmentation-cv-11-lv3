import mlflow
import mlflow.pytorch
import os
class MLflowManager:
    def __init__(self,tracking_uri='https://movie-pregnancy-robert-addressed.trycloudflare.com',experiment_name="default"):
        '''
        uri트래킹을 알아서 하게 설정, 실험 명 설정 가능
        Args: tracking_uri(str) : 트래킹 uri ,
              experiment_name(str) : 실험 명 
        '''
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def start_run(self,run_name=None):  
        '''
        런 네임 생성 및 반환
        Args: run_namge(str) : 사용할 런네임 
        '''
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self,params):
        '''
        파라미터 기록
        Args: params(dict) : 기록할 파라미터 딕셔너리

        '''
        mlflow.log_params(params)

    def log_metrics(self,metrics,step=None):
        '''
        메트릭 값 기록
        
        Args :
        metrics(dict) : 기록할 메트릭
        step(int) : 메트릭를 위한 스텝
        '''
        for key,value in metrics.items():
            mlflow.log_metric(key,value,step=step)

    def log_model(self,model,model_dir='model'):
        '''
        pt파일 저장
        Args : model(model) : 저장할 모델
              model_dir(str) : 저장할 파일 디렉토리 이름
        '''
        mlflow.pytorch.log_model(model,model_dir)
    def log_artifacts(self,dir_path):
        '''
        디렉토리 내 모든 파일을 Mlflow에 업로드
        Args: dir_path(str) : 업로드할 디렉토리 경로
        '''
        if os.path.exists(dir_path):
            mlflow.log_artifacts(dir_path)
            