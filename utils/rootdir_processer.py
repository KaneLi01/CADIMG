import os, random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Iterator
import multiprocessing
from multiprocessing.context import TimeoutError
import traceback


class FileProcessor(ABC):
    """根据目录级别批量处理文件的抽象基类"""
    
    def __init__(self, root_dir: str, extension: str, depth: int = 2, 
                 output_root_dir: Optional[str] = None):
        """
        初始化文件处理器
        
        Args:
            root_dir: 根目录路径
            extension: 要处理的文件扩展名（如 '.txt'）
            depth: 遍历深度（1: 根目录文件，2: 一级子目录文件，3: 二级子目录文件）
            output_root_dir: 输出根目录（可选）
        """
        self.root_dir = os.path.normpath(root_dir)
        self.extension = extension.lower()
        self.depth = depth
        self.output_root_dir = os.path.normpath(output_root_dir) if output_root_dir else None
        
        # 验证目录存在
        if not os.path.exists(self.root_dir):
            raise ValueError(f"根目录不存在: {self.root_dir}")
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"根目录不是目录: {self.root_dir}")
        
        # 验证深度值
        if depth < 1:
            raise ValueError("深度值必须至少为1")

    def iter_files(self) -> Iterator[str]:
        """
        遍历指定深度的所有文件
        
        Yields:
            符合条件的文件完整路径
        """
        if self.depth == 1:
            # 处理根目录下的文件
            yield from self._iter_directory_files(self.root_dir)
            
        elif self.depth == 2:
            # 处理一级子目录下的文件
            for subdir in self._get_subdirectories(self.root_dir):
                yield from self._iter_directory_files(subdir)
                
        elif self.depth == 3:
            # 处理二级子目录下的文件
            for first_level_dir in self._get_subdirectories(self.root_dir):
                for second_level_dir in self._get_subdirectories(first_level_dir):
                    yield from self._iter_directory_files(second_level_dir)
                    
        else:
            raise ValueError(f"不支持的深度值: {self.depth}，目前支持1-3")

    def _get_subdirectories(self, directory: str) -> Iterator[str]:
        """获取目录下的所有子目录（按目录名排序）"""
        try:
            entries = list(os.scandir(directory))
            entries.sort(key=lambda x: x.name)  # 按名称排序
            
            for entry in entries:
                if entry.is_dir():
                    yield entry.path
        except (OSError, PermissionError) as e:
            print(f"无法访问目录 {directory}: {e}")

    def _iter_directory_files(self, directory: str) -> Iterator[str]:
        """遍历目录下的所有文件，过滤出符合条件的文件（按文件名排序）"""
        try:
            entries = list(os.scandir(directory))
            entries.sort(key=lambda x: x.name)  # 按名称排序
            
            for entry in entries:
                if (entry.is_file() and 
                    self._should_process(entry.path)):
                    yield entry.path
        except (OSError, PermissionError) as e:
            print(f"无法访问目录 {directory}: {e}")

    def _should_process(self, filepath: str) -> bool:
        """检查文件是否应该被处理"""
        return (os.path.splitext(filepath)[1].lower() == self.extension and
                os.path.isfile(filepath))

    def _get_output_path(self, input_file: str) -> str:
        """根据输入文件路径生成输出路径"""
        rel_path = os.path.relpath(input_file, self.root_dir)
        return os.path.join(self.output_root_dir, rel_path)

    def print_all_files(self) -> None:
        """打印所有符合条件的文件（用于验证逻辑）"""
        print(f"根目录: {self.root_dir}")
        print(f"扩展名: {self.extension}")
        print(f"深度: {self.depth}")
        print("找到的文件:")
        print("-" * 50)
        
        file_count = 0
        for filepath in self.iter_files():
            print(filepath)
            file_count += 1
            
        print("-" * 50)
        print(f"总共找到 {file_count} 个文件")

    def process_all(self, timeout: int = 15, processes: int = 1) -> None:
        """
        使用多进程处理所有文件
        
        Args:
            timeout: 单个文件处理超时时间（秒）
            processes: 进程池大小
        """
        file_count = 0
        success_count = 0
        timeout_count = 0
        error_count = 0
        
        print(f"开始处理文件，超时时间: {timeout}秒，进程数: {processes}")
        print("-" * 60)
        
        with multiprocessing.Pool(processes=processes) as pool:
            results = []
            
            # 提交所有任务到进程池
            for filepath in self.iter_files():
                output_path = self._get_output_path(filepath) if self.output_root_dir else None
                file_count += 1
                
                # 提交任务到进程池
                result = pool.apply_async(
                    self._process_file_wrapper, 
                    (filepath, output_path)
                )
                results.append((filepath, result))
            
            # 收集结果
            for filepath, result in results:
                try:
                    # 等待结果，设置超时
                    result.get(timeout=timeout)
                    success_count += 1
                    print(f"[成功处理] {filepath}")
                    
                except TimeoutError:
                    timeout_count += 1
                    print(f"[超时跳过] {filepath} (超过{timeout}秒)")
                    
                except Exception as e:
                    error_count += 1
                    print(f"[错误跳过] {filepath} 报错: {e}")
                    traceback.print_exc()
        
        # 输出统计信息
        print("-" * 60)
        print(f"处理完成！统计信息:")
        print(f"总文件数: {file_count}")
        print(f"成功处理: {success_count}")
        print(f"超时跳过: {timeout_count}")
        print(f"错误跳过: {error_count}")

    def _process_file_wrapper(self, input_filepath: str, output_path: Optional[str] = None) -> None:
        """
        包装process_file方法，用于多进程环境
        
        注意：这个方法会在子进程中运行，确保process_file方法是线程安全的
        """
        try:
            self.process_file(input_filepath, output_path)
        except Exception as e:
            # 重新抛出异常以便在主进程中捕获
            raise e

    def process_file(self, input_filepath: str, output_path: Optional[str] = None, **kwargs) -> None:
        """处理单个文件的抽象方法（由子类实现）"""
        pass

    def check_subdirs_num(self, dir_path: str = None, n: int = 6, mode: str = 'check') -> bool:
        """
        根据depth检查对应层级的文件数量是否满足要求（只检查当前层级文件）
        
        Args:
            dir_path: 要检查的目录路径，默认为self.root_dir
            n: 期望的文件数量。如果n=0，检查空目录；如果n!=0，检查文件数不等于n的目录
            mode: 'check' 只检查并打印，'del' 检查并删除不符合条件的目录
        
        Returns:
            bool: 如果所有目录都符合条件返回True，否则返回False
        """
        import shutil
        
        target_dir = dir_path if dir_path else self.root_dir
        all_valid = True
        
        # 根据depth确定要检查的目录层级
        if self.depth == 1:
            # depth=1: 检查根目录下的文件数量
            directories_to_check = [target_dir]
            
        elif self.depth == 2:
            # depth=2: 检查一级子目录
            directories_to_check = list(self._get_subdirectories(target_dir))
            
        elif self.depth == 3:
            # depth=3: 检查二级子目录
            directories_to_check = []
            for first_level_dir in self._get_subdirectories(target_dir):
                directories_to_check.extend(self._get_subdirectories(first_level_dir))
                
        else:
            raise ValueError(f"不支持的深度值: {self.depth}")
        
        print(f"检查 {len(directories_to_check)} 个目录，期望文件数: {n}")
        
        for dir_path in directories_to_check:
            try:
                # 只检查当前目录下的文件（不包括子目录）
                file_count = 0
                for entry in os.scandir(dir_path):
                    if entry.is_file():
                        file_count += 1
                
                should_process = False
                message = ""
                
                if n == 0:
                    # 检查空目录
                    if file_count == 0:
                        should_process = True
                        message = f"{dir_path} 目录下没有文件"
                else:
                    # 检查文件数不等于n的目录
                    if file_count != n:
                        should_process = True
                        message = f"{dir_path} 目录下有{file_count}个文件，期望{n}个"
                
                if should_process:
                    all_valid = False
                    if mode == 'check':
                        print(f"[检查] {message}")
                    elif mode == 'del':
                        print(f"[删除] {message}")
                        if file_count == 0:
                            os.rmdir(dir_path)  # 删除空目录
                        else:
                            shutil.rmtree(dir_path)  # 删除非空目录
                    else:
                        raise ValueError(f"不支持的模式: {mode}，请使用 'check' 或 'del'")
                        
            except (OSError, PermissionError) as e:
                print(f"无法处理目录 {dir_path}: {e}")
                all_valid = False
        
        if all_valid:
            print(f"所有目录都满足要求（期望文件数: {n}）")
        else:
            print(f"有目录不满足要求（期望文件数: {n}）")
        
        return all_valid

