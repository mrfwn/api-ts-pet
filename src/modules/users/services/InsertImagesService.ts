import { injectable, inject } from 'tsyringe';

import AppError from '@shared/errors/AppError';

import IUsersRepositoy from '@modules/users/repositories/IUsersRepository';
import IImagesRepository from '@modules/users/repositories/IImagesRepository';

import IStorageProvider from '@shared/container/providers/StorageProvider/models/IStorageProvider';
import Image from '../infra/typeorm/entities/Image';

interface IRequest {
  user_id: string;
  filesName: string[];
}

@injectable()
class InsertImageService {
  constructor(
    @inject('UsersRepository')
    private usersRepository: IUsersRepositoy,

    @inject('ImagesRepository')
    private imagesRepository: IImagesRepository,

    @inject('StorageProvider')
    private storageProvider: IStorageProvider,
  ) {}

  public async execute(data: IRequest): Promise<Image[]> {
    const checkUserExists = await this.usersRepository.findById(data.user_id);

    if (!checkUserExists) {
      throw new AppError('User does not exist');
    }

    if (data.filesName) {
      await this.storageProvider.saveListFile(data.filesName);
      const fileList = data.filesName.map(name => {
        return {
          user_id: data.user_id,
          name,
        };
      });
      return this.imagesRepository.insertImages(fileList);
    }
    return [];
  }
}

export default InsertImageService;
