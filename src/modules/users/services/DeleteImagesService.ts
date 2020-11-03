import { injectable, inject } from 'tsyringe';

import AppError from '@shared/errors/AppError';

import IUsersRepository from '@modules/users/repositories/IUsersRepository';

import IStorageProvider from '@shared/container/providers/StorageProvider/models/IStorageProvider';

import IImagesRepository from '../repositories/IImagesRepository';

interface IRequest {
  user_id: string;
  filesName: string[];
}
@injectable()
class CreateCaseStudentService {
  constructor(
    @inject('UsersRepository')
    private usersRepository: IUsersRepository,

    @inject('ImagesRepository')
    private imagesRepository: IImagesRepository,

    @inject('StorageProvider')
    private storageProvider: IStorageProvider,
  ) {}

  public async execute({ user_id, filesName }: IRequest): Promise<void> {
    const checkIfUserExist = await this.usersRepository.findById(user_id);

    if (!checkIfUserExist) {
      throw new AppError('User not found');
    }

    await this.imagesRepository.deleteImages(filesName);

    // await this.storageProvider.deleteListFile(listFiles);

    // await this.casesStudentsRepository.delete(case_id);
  }
}

export default CreateCaseStudentService;
