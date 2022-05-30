from argparse import Namespace, ArgumentParser

import pandas as pd
import requests


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str)
    parser.add_argument('-p', '--port', required=True, type=int)
    parser.add_argument('-n', '--n_requests', required=True, type=int)
    return parser.parse_args()


def main(args: Namespace):
    df = pd.read_csv(args.file, index_col=0)
    for i in range(args.n_requests):
        request_data = df.iloc[i:i+1].to_dict(orient='list')
        print(f'request_data = {request_data}')
        response = requests.post(
            f'http://0.0.0.0:{args.port}/predict',
            headers={
                'Content-Type': 'application/json',
                'accept': 'application/json',
            },
            json=request_data
        )
        print(f'status_code = {response.status_code}')
        print(f'response.json = {response.json()}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
