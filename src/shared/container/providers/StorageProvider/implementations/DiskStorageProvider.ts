import fs from 'fs';
import path from 'path';
import uploadConfig from '@config/upload';
import IStorageProvider from '../models/IStorageProvider';

class DiskStorageProvider implements IStorageProvider {
  public async saveFile(file: string): Promise<string> {
    await fs.promises.rename(
      path.resolve(uploadConfig.tmpFolder, file),
      path.resolve(uploadConfig.uploadsFolder, file),
    );

    return file;
  }

  public async saveListFile(listFile: string[]): Promise<void> {
    await Promise.all(
      listFile.map(async file =>
        fs.promises.rename(
          path.resolve(uploadConfig.tmpFolder, file),
          path.resolve(uploadConfig.uploadsFolder, file),
        ),
      ),
    );
  }

  public async deleteFile(file: string): Promise<void> {
    const filePath = path.resolve(uploadConfig.uploadsFolder, file);

    try {
      await fs.promises.stat(filePath);
    } catch {
      return;
    }

    await fs.promises.unlink(filePath);
  }

  public async deleteListFile(listFile: string[]): Promise<void> {
    const listFilePath = listFile.map(file =>
      path.resolve(uploadConfig.uploadsFolder, file),
    );

    try {
      await Promise.all(
        listFilePath.map(async filePath => fs.promises.stat(filePath)),
      );
    } catch {
      return;
    }
    await Promise.all(
      listFilePath.map(async filePath => fs.promises.unlink(filePath)),
    );
  }
}

export default DiskStorageProvider;
