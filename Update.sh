#Factor
python updata.py ./checkpoint/factor0.1_99.pth.tar -m Factor  -n factor01_99 -d updatedCheckpoint
python updata.py ./checkpoint/Factor0.05_49.pth.tar -m Factor  -n Factor005_49 -d updatedCheckpoint
python updata.py ./checkpoint/Factor0.025_46.pth.tar -m Factor  -n Factor0025_46 -d updatedCheckpoint
python updata.py ./checkpoint/Factor0.01_49.pth.tar -m Factor  -n Factor001_49 -d updatedCheckpoint
python updata.py ./checkpoint/Factor0.005_49.pth.tar -m Factor  -n Factor0005_49 -d updatedCheckpoint
python updata.py ./checkpoint/Factor0.001_49.pth.tar -m Factor  -n Factor0001_49 -d updatedCheckpoint
#Hyper
python updata.py ./checkpoint/hyper0.1_99.pth.tar -m Hyper  -n Hyper01_99 -d updatedCheckpoint
python updata.py ./checkpoint/Hyper0.05_48.pth.tar -m Hyper  -n Hyper005_48 -d updatedCheckpoint
python updata.py ./checkpoint/Hyper0.025_46.pth.tar -m Hyper  -n Hyper0025_46 -d updatedCheckpoint
python updata.py ./checkpoint/Hyper0.01_47.pth.tar -m Hyper  -n Hyper001_47 -d updatedCheckpoint
python updata.py ./checkpoint/Hyper0.005_47.pth.tar -m Hyper  -n Hyper0005_47 -d updatedCheckpoint
python updata.py ./checkpoint/Hyper0.001_48.pth.tar -m Hyper  -n Hyper0001_48 -d updatedCheckpoint
#Joint
python updata.py ./checkpoint/Joint0.1_49.pth.tar -m Joint  -n Joint01_49 -d updatedCheckpoint
python updata.py ./checkpoint/Joint0.05_49.pth.tar -m Joint  -n Joint005_49 -d updatedCheckpoint
python updata.py ./checkpoint/Joint0.025_49.pth.tar -m Joint  -n Joint0025_49 -d updatedCheckpoint
python updata.py ./checkpoint/Joint0.01_49.pth.tar -m Joint  -n Joint001_49 -d updatedCheckpoint
python updata.py ./checkpoint/Joint0.005_49.pth.tar -m Joint  -n Joint0005_49 -d updatedCheckpoint
python updata.py ./checkpoint/Joint0.001_46.pth.tar -m Joint  -n Joint0001_46 -d updatedCheckpoint
#Cheng2020
python updata.py ./checkpoint/Cheng2020Attention0.1_48.pth.tar -m Cheng2020Attention  -n Cheng2020Attention01_48 -d updatedCheckpoint
python updata.py ./checkpoint/Cheng2020Attention0.05_49.pth.tar -m Cheng2020Attention  -n Cheng2020Attention005_49 -d updatedCheckpoint
python updata.py ./checkpoint/Cheng2020Attention0.025_49.pth.tar -m Cheng2020Attention  -n Cheng2020Attention0025_49 -d updatedCheckpoint
python updata.py ./checkpoint/ChengAttention0.01_49.pth.tar -m Cheng2020Attention  -n ChengAttention001_49 -d updatedCheckpoint
python updata.py ./checkpoint/Cheng2020Attention0.005_48.pth.tar -m Cheng2020Attention  -n Cheng2020Attention0005_48 -d updatedCheckpoint
python updata.py ./checkpoint/Cheng2020Attention0.001_48.pth.tar -m Cheng2020Attention  -n Cheng2020Attention0001_48 -d updatedCheckpoint
#FactorGlobalModule
python updata.py ./checkpoint/FactorTrans0.1.pth.tar -m FactorGlobalModule  -n FactorTrans01 -d updatedCheckpoint
python updata.py ./checkpoint/FactorTrans0.05_49.pth.tar -m FactorGlobalModule  -n FactorTrans005_49 -d updatedCheckpoint
python updata.py ./checkpoint/FactorTrans0.025_48.pth.tar -m FactorGlobalModule  -n FactorTrans0025_48 -d updatedCheckpoint
python updata.py ./checkpoint/FactorTrans0.01_48.pth.tar -m FactorGlobalModule  -n FactorTrans001_48 -d updatedCheckpoint
python updata.py ./checkpoint/FactorTrans0.005_48.pth.tar -m FactorGlobalModule  -n FactorTrans0005_48 -d updatedCheckpoint
python updata.py ./checkpoint/FactorTrans0.001_44.pth.tar -m FactorGlobalModule  -n FactorTrans0001_44 -d updatedCheckpoint
#HyperGlobalModule
python updata.py ./checkpoint/HyperTrans0.1.pth.tar -m HyperGlobalModule  -n HyperTrans01 -d updatedCheckpoint
python updata.py ./checkpoint/HyperTrans0.05_46.pth.tar -m HyperGlobalModule  -n HyperTrans005_46 -d updatedCheckpoint
python updata.py ./checkpoint/HyperTrans0.025_46.pth.tar -m HyperGlobalModule  -n HyperTrans0025_46 -d updatedCheckpoint
python updata.py ./checkpoint/HyperTrans0.01_48.pth.tar -m HyperGlobalModule  -n HyperTrans001_48 -d updatedCheckpoint
python updata.py ./checkpoint/HyperTrans0.005_48.pth.tar -m HyperGlobalModule  -n HyperTrans0005_48 -d updatedCheckpoint
python updata.py ./checkpoint/HyperTrans0.001_34.pth.tar -m HyperGlobalModule  -n HyperTrans0001_34 -d updatedCheckpoint
#BaseModel
python updata.py ./checkpoint/Basemodel0.1_44.pth.tar -m ProposedBasemodel  -n Basemodel01_44 -d updatedCheckpoint
python updata.py ./checkpoint/Basemodel0.05_49.pth.tar -m ProposedBasemodel  -n Basemodel005_49 -d updatedCheckpoint
python updata.py ./checkpoint/Basemodel0.025_49.pth.tar -m ProposedBasemodel  -n Basemodel0025_49 -d updatedCheckpoint
python updata.py ./checkpoint/Basemodel0.01_43.pth.tar -m ProposedBasemodel  -n Basemodel001_43 -d updatedCheckpoint
python updata.py ./checkpoint/Basemodel0.005_46.pth.tar -m ProposedBasemodel  -n Basemodel0005_46 -d updatedCheckpoint
python updata.py ./checkpoint/Basemodel0.001_49.pth.tar -m ProposedBasemodel  -n Basemodel0001_49 -d updatedCheckpoint
#ProposedModule
python updata.py ./checkpoint/PLConvTrans0.1.pth.tar -m Proposed  -n PLConvTrans01 -d updatedCheckpoint
python updata.py ./checkpoint/PLConvTrans0.05.pth.tar -m Proposed  -n PLConvTrans005 -d updatedCheckpoint
python updata.py ./checkpoint/PLConvTrans0.025_48.pth.tar -m Proposed  -n PLConvTrans0025_48 -d updatedCheckpoint
python updata.py ./checkpoint/PLConvTrans0.01.pth.tar -m Proposed  -n PLConvTrans001 -d updatedCheckpoint
python updata.py ./checkpoint/PLConvTrans0.005.pth.tar -m Proposed  -n PLConvTrans0005 -d updatedCheckpoint
python updata.py ./checkpoint/PLConvTrans0.001.pth.tar -m Proposed  -n PLConvTrans0001 -d updatedCheckpoint
