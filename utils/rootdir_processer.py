import os, random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import multiprocessing
from multiprocessing.context import TimeoutError
import traceback


class FileProcessor(ABC):
    def __init__(self, root_dir: str, extension: str, output_root_dir: Optional[str] = None):
        """
        root_dir: 输入根目录路径
        extension: 最底层文件后缀名(需要带'.'，如'.txt')
        output_root_dir: 输出根目录路径，用于保存
        """
        self.root_dir = os.path.normpath(root_dir)  
        self.extension = extension.lower()
        self.output_root_dir = os.path.normpath(output_root_dir) if output_root_dir else None

    def process_all(self, conti=None, timeout: int = 15) -> None:
        for first_level in sorted(os.scandir(self.root_dir), key=lambda x: x.name):
            if not first_level.is_dir():
                continue
            for second_level in sorted(os.scandir(first_level.path), key=lambda x: x.name):
                if not second_level.is_dir():
                    continue
                if conti:
                    if str(second_level.name) < conti:
                        continue
                for entry in sorted(os.scandir(second_level.path), key=lambda x: x.name):
                    if entry.is_file() and self._should_process(entry.path):
                        output_path = self._get_output_path(entry.path) if self.output_root_dir else None
                        # === 用多进程池执行 process_file，防止超时卡死 ===
                        with multiprocessing.Pool(processes=1) as pool:
                            try:
                                result = pool.apply_async(self.process_file, (entry.path, output_path))
                                result.get(timeout=timeout)
                            except TimeoutError:
                                print(f"[超时跳过] {entry.path}")
                            except Exception as e:
                                print(f"[错误跳过] {entry.path} 报错: {e}")
                                traceback.print_exc()
            print(f'directory {first_level.name} has been processed')

    def get_random_file(self) -> Optional[str]:
        """
        随机获取一个最后一级子目录下的随机一个文件
        返回: 文件的完整路径，如果没有找到符合条件的文件则返回None
        """
        all_files = []
        
        # 遍历所有最后一级子目录（second_level）
        for first_level in os.scandir(self.root_dir):
            if not first_level.is_dir():
                continue
            for second_level in os.scandir(first_level.path):
                if not second_level.is_dir():
                    continue
                # 收集该目录下所有符合后缀条件的文件
                for entry in os.scandir(second_level.path):
                    if entry.is_file() and self._should_process(entry.path):
                        all_files.append(entry.path)
        
        # 如果没有找到任何文件，返回None
        if not all_files:
            return None
        
        final_file = random.choice(all_files)
        print('处理文件：', final_file)

        # 随机选择一个文件
        return final_file

    def get_a_file(self, n=0) -> Optional[str]:
        """
        按照排序，获取最后一级子目录下的第1个文件
        返回: 文件的完整路径，如果没有找到符合条件的文件则返回None
        """
        # 按字母顺序遍历所有最后一级子目录
        for first_level in sorted(os.scandir(self.root_dir), key=lambda x: x.name):
            if not first_level.is_dir():
                continue
            for second_level in sorted(os.scandir(first_level.path), key=lambda x: x.name):
                if not second_level.is_dir():
                    continue
                # 获取该目录下所有符合条件的文件并排序
                files = []
                for entry in sorted(os.scandir(second_level.path), key=lambda x: x.name):
                    if entry.is_file() and self._should_process(entry.path):
                        files.append(entry.path)
                
                if files:
                    print(f'处理排第{n}个的文件：{files[n]}')
                    return files[n]
        
        

        # 如果没有找到任何文件，返回None
        return None

    def find_del_empty_dirs(self, dele=False) -> List[str]:
        """查找没有文件的目录，并选择删除"""
        empty_dirs = []
        
        for dirpath, dirnames, filenames in os.walk(self.root_dir, topdown=False):  # 自底向上遍历
            if not dirnames and not any(f.lower().endswith(self.extension) for f in filenames):
                empty_dirs.append(dirpath)
        
        # 如果需要删除
        if dele:
            for dir_path in sorted(empty_dirs, key=len, reverse=True):  # 按路径长度倒序（先删最深目录）
                try:
                    os.rmdir(dir_path)
                    print(f"已删除空目录: {dir_path}")
                except OSError as e:
                    print(f"删除目录失败 {dir_path}: {e}")
        
        return empty_dirs

    def count_first_level_subdirs(self) -> Tuple[List[str], int]:
        """统计第一级子目录"""
        subdirs = []
        with os.scandir(self.root_dir) as it:
            for entry in it:
                if entry.is_dir():
                    subdirs.append(entry.name)
        return sorted(subdirs), len(subdirs)

    def count_second_level_subdirs(self) -> Dict[str, int]:
        """统计每个二级子目录"""
        result = {}
        # 先获取并排序第一级目录
        for first_level in sorted(
            [d for d in os.scandir(self.root_dir) if d.is_dir()],
            key=lambda x: x.name
        ):
            result[first_level.name] = sum(
                1 for entry in os.scandir(first_level.path) if entry.is_dir()
            )
        return result
    
    def list_all_second_level_subdirs(self) -> List[str]:
        """
        获取所有二级子目录的名称，返回列表
        """
        second_level_dirs = []
        for first_level in sorted(
            [d for d in os.scandir(self.root_dir) if d.is_dir()],
            key=lambda x: x.name
        ):
            for second_level in sorted(
                [d for d in os.scandir(first_level.path) if d.is_dir()],
                key=lambda x: x.name
            ):
                second_level_dirs.append(second_level.name)
        return second_level_dirs

    def _should_process(self, filepath: str) -> bool:
        """
        根据后缀判断是否应该处理该文件
        """
        return os.path.splitext(filepath)[1].lower() == self.extension
    
    def _get_output_path(self, input_file: str) -> str:
        """
        如果有输出根目录，则根据当前文件相对路径来获取输出文件路径
        """
        rel_path = os.path.relpath(input_file, self.root_dir)
        output_path = os.path.join(self.output_root_dir, rel_path)
        return output_path
    
    def process_file(self, input_filepath: str, output_path: Optional[str] = None, **kwargs) -> None:
        """
        处理单个文件的抽象方法
        """
        pass
