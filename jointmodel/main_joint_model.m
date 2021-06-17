load obs;load shortbaseline;
load geo_tcp;load height_tcp;
obs(:,3:4)=geo_tcp;obs(:,5)=height_tcp;
Output=jointmodel(obs,shortbaseline,80000,9.6000000e+09,667425.0045,760,640,20.1048,11.18145476827794302475,11.313365,300,300,700,4);
