import polymath.polymath as pm
from polymath.polymath.srdfg.passes.compiler_passes import  Lower
from axiline.axiline.run_passes import VerilogGenerateFixedBitwidth,VerilogGenerateFlexBitwidth
from math import ceil,log2
import argparse,json
import os.path

CONFIG_DIR="./verilog_config.json"

class AxilineCompiler:
    def __init__(self,benchmark,config,dimension=[]):
        self.benchmark=benchmark
        self.config=config
        self.dimension=dimension
        self.param={}
        self.vh = ""

    @property
    def benchmark(self):
        return self._benchmark

    @benchmark.setter
    def benchmark(self, benchmark):
        if isinstance(benchmark,str):
            if ( benchmark in ["svm", "reco", "logistic", "linear"] ):
                self._benchmark = benchmark
            elif os.path.isfile(benchmark):
                self._benchmark = benchmark
        else:
            print("benchmark error")

    @property
    def dimension(self):
        return self._dimension

    @dimension.setter
    def dimension(self,dim):
        if (isinstance(dim,int)) :
            self._dimension=[dim]
        elif isinstance(dim, list) and all(isinstance(x, int) for x in dim):
            self._dimension=dim
        else:
            print("dimension error")

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        if (set(["algo", "input_bitwidth","fixed_bitwidth", "param_list"]).issubset(set(config.keys()))):
            self._config = config
        else:
            print("config error")

    def compile_with_lower_graph(self,onnx_path,path_to_output):
        graph = pm.from_onnx(onnx_path)
        shapes = {}
        shape_pass = pm.NormalizeGraph(shapes)
        transformed_graph = shape_pass(graph)
        lower_pass = Lower({})
        lowered_graph = lower_pass(transformed_graph, {})
        if (self.config["fixed_bitwidth"]):
            verilog_pass =VerilogGenerateFixedBitwidth(self.config)
        else:
            verilog_pass = VerilogGenerateFlexBitwidth(self.config)
        # # file path of
        graph = verilog_pass(lowered_graph)
        verilog_pass.create_verilog_nonembedded(path_to_output)

    def compiler_with_benchmark_dimension(self):
        if not self.fold:
            exit("folding parameter is not defined!")
        benchmark = self.benchmark
        if len(self.dimension)==1 and isinstance(self.dimension[0],int):
            # parameters for SVM, LinearRegression and logisticRegression
            if (benchmark =="svm" or benchmark =="linear" or benchmark =="logstic"):
                self.param["SIZE"]=ceil(self.dimension[0]/self.fold)
                self.param["NUM_CYCLE"]=self.fold
                self.param["LOG_NUM_CYCLE"] =ceil(log2(self.fold))
                self.param["INPUT_BITWIDTH"]=self.config["input_bitwidth"]
                self.param["BITWIDTH"] =self.config["bitwidth"]
                self.param["NUM_UNIT"]=1
                self.param[benchmark.upper()]=1
            else:
                exit("benchmark is not supported!")
        # parameters for RecommenderSystems
        elif len(self.dimension)==3:
            m,n,k=self.dimension
            if self.benchmark=="reco":
                self.param["SIZE"] = k
                self.param["NUM_CYCLE"] = 1
                self.param["LOG_NUM_CYCLE"] = 1
                self.param["INPUT_BITWIDTH"] = self.config["input_bitwidth"]
                self.param["BITWIDTH"] = self.config["bitwidth"]
                self.param["NUM_UNIT"] = ceil(m + n / self.fold)
                self.param[benchmark.upper()] = 1
        else:
            exit("benchmark dimension error!")

    # generate verilog head file to path
    def generate_vh(self,path_to_vh=0):
        vh=""
        for key in self.param.keys():
            vh+=f"`define   {key} {str(self.param[key])}\n"
        if(path_to_vh):
            file = open(path_to_vh, 'w')
            file.write(vh)
            file.close()
        else:
            return vh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Axiline compile Framework")
    # generate pipeline RTL or Combinational RTL
    parser.add_argument("--benchmark", "-b", required=True,
                        help="select a benchmark, 'svm', 'logistic', 'linear', 'reco', or a onnx file directory")
    parser.add_argument("--dimension", "-d", nargs='+', required=False,
                        help="select a feature size (optional)")
    parser.add_argument("--config", "-c", required=False,
                        help="choose a json config, or use default")
    parser.add_argument("--output", "-o", required=True,
                        help="output directory")
    args = parser.parse_args()

    # check config
    if ( hasattr(args, 'config') and os.path.isfile(args.config)):
        with open(args.config) as f:
            config = json.load(f)
        f.close()
    else:
        print("Use default Verilog config!")
        with open(CONFIG_DIR) as f:
            config = json.load(f)
        f.close()

    # check benchmark
    if(args.benchmark=="svm" or args.benchmark=="logistic" or args.benchmark=="linear" or args.benchmark=="reco" ):
            compiler = AxilineCompiler(args.benchmark,config,args.dimension)
            compiler.compiler_with_benchmark_dimension
            compiler.generate_vh(args.output)

    elif(os.path.isfile(args.benchmark)):
        compiler = AxilineCompiler(args.benchmark, config)
        compiler.compile_with_lower_graph(args.benchmark,args.output)



