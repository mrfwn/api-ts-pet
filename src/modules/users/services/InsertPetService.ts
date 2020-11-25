import { injectable, inject } from 'tsyringe';
import path from 'path';
import { spawn } from 'promisify-child-process';
import AppError from '@shared/errors/AppError';
import fs from 'fs';

import IUsersRepositoy from '@modules/users/repositories/IUsersRepository';
import IImagesRepository from '@modules/users/repositories/IImagesRepository';

import IStorageProvider from '@shared/container/providers/StorageProvider/models/IStorageProvider';
import Image from '../infra/typeorm/entities/Image';

interface IRequest {
  user_id: string;
  imagesIds: string[];
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

  public async execute(data: IRequest): Promise<Image> {
    const { user_id, imagesIds } = data;
    const checkUserExists = await this.usersRepository.findById(user_id);
    if (!checkUserExists) {
      throw new AppError('User does not exist');
    }

    const script = path.resolve(
      __dirname,
      '..',
      '..',
      '..',
      'stargan',
      'main.py',
    );

    fs.writeFile('imgNames.txt', imagesIds[0], 'utf-8', () => {});

    await spawn('python', [script], {
      encoding: 'utf8',
    });

    const name = fs.readFileSync('output.txt', 'utf8');

    await this.storageProvider.saveFile(name);

    return this.imagesRepository.insertPet({
      user_id,
      name,
      type: true,
    });

    return new Image();
  }
}

export default InsertImageService;
