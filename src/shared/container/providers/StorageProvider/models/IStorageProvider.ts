export default interface IStorageProvider {
  saveFile(file: string): Promise<string>;
  saveListFile(listFile: string[]): Promise<void>;
  deleteFile(file: string): Promise<void>;
  deleteListFile(listFile: string[]): Promise<void>;
}
