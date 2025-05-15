# algoTrading

Setup 
git clone https://github.com/Liorbardak/algoTraing.git
- prepare data directory - see example for data directory 
- Run (one time)  utils/preproc.py
  - Create reference index - the average of all stocks
  - filter stocks with missing dates  

Run trading simulation
sim.py 

On terminal 
cd C:\Users\dadab\projects\algotrading\algotrading_bot
#.venv\Scripts\activate
.venv2\Scripts\activate

update code and run :
python  basic_code\sim.py
or : 
python  basic_code\sim_runner.py -i <inputdir> -o <output_dir> -b <bot list>

Training
Predictor 
- Run prepare_data_for_training.py for Create the training/validation/test data
- run training:  python  ML\Predictor\run_training.py
- run trained predictor , create data frame with predictions -  python  ML\Predictor\run_inference.py
      

DB : 
v4 - with extra features
v5 - with extra features + 200 stocks outof s&p 