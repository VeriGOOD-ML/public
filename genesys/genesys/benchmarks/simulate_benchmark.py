
def simulate_benchmark(benchmark_name):
    pass

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='GeneSys compiler')
    argparser.add_argument('-b', '--benchmark', required=True,
                           help='Name of the benchmark to create. One of "resnet18", "resnet50", or "maskrcnn".')

    args = argparser.parse_args()
    simulate_benchmark(args.benchmark)
