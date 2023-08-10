#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/7/30 21:45
@File    : clear_exp_results.py
@Software: PyCharm
@Desc    :
"""
from functools import partial
from pathlib import Path

import click
from alchemy_cat.py_tools import yprint, gprint, Logger, get_local_time_str
from tqdm import tqdm

kPattern = '**/cam'


def white_filter(p: Path, wl: tuple[str]) -> bool:
    p = str(p)
    for ws in wl:
        if ws in p:
            return False
    return True


@click.command()
@click.option('-r', '--root', type=str, help="根目录。")
@click.option('--avs', default=100, type=float, help="平均文件大小(KB)。")
@click.option('--ts', default=1000, type=float, help="总文件大小（MB）。")
@click.option('--simulate', default=True, type=bool, help="是否模拟删除。")
@click.argument("wl", nargs=-1, type=str)
def main(root: str, avs: float, ts: float, simulate: bool, wl: tuple[str]):
    # * 参数解析。
    root = Path(root)
    assert root.exists(), f"根目录{root}不存在。"
    # ** 文件大小单位转换为字节。
    avs *= 1024
    ts *= 1024 * 1024

    print(f"""
>>> 开始从{str(root)}中搜索名为{kPattern}的目录。
>>> 若目录中的文件平均大小大于{avs / 1024}KB，且总大小大于{ts / 1024 ** 2}MB，则删除该目录下所有文件。
>>> 路径若匹配白名单，则跳过。白名单为：
>>> {wl}
>>> 模拟删除：{simulate}。
    """)

    # * 找到所有cam目录。
    cam_dirs = [p.resolve(strict=True) for p in root.glob(kPattern)]  # 理论上会忽略所有软连接。

    # * 过滤白名单。
    cam_dirs = list(filter(partial(white_filter, wl=wl), cam_dirs))

    # * 遍历所有cam目录。
    for cam_dir in cam_dirs:
        # * 找到所有npz文件。
        npz_files = list(cam_dir.glob('*.npz'))

        # * 若目录下没有npz文件，则跳过。
        if len(npz_files) == 0:
            continue

        # * 确定npz文件的总大小和平均大小。
        npz_sizes = [npz_file.stat().st_size for npz_file in npz_files]
        npz_total_size = sum(npz_sizes)
        npz_av_size = npz_total_size / len(npz_sizes)

        # * 若总大小大于ts且平均大小大于avs，则删除该目录下所有文件。
        print(f"{str(cam_dir)} 下的一共有 {len(npz_files)} 个npz文件：\n"
              f"总大小为 {npz_total_size // 1024 ** 2} MB；\n"
              f"平均大小为 {npz_av_size // 1024} KB。")

        if npz_total_size > ts and npz_av_size > avs:
            if simulate:
                gprint(f"**模拟**删除中...")
            else:
                gprint(f"删除中...")
                for npz_file in tqdm(npz_files, desc="删除", unit="个", dynamic_ncols=True):
                    npz_file.unlink(missing_ok=False)
                # subprocess.run(f'rm {str(cam_dir / "*.npz")}', shell=True)  # 系统限制*能匹配的文件数量。
                # ** 留下删除记录。
                (cam_dir / '.CAM已清理').write_text(f"清理于{get_local_time_str()}。")
        else:
            yprint(f"跳过 {str(cam_dir)}。")

        print("------------------------------------------------------", end="\n\n")


if __name__ == '__main__':
    Logger(f'temp/清理CAM日志/{get_local_time_str(for_file_name=True)}.log')
    main()
