import { injectable, inject } from 'tsyringe';

import AppError from '@shared/errors/AppError';

import IUsersRepository from '@modules/users/repositories/IUsersRepository';
import IImagesRepository from '@modules/users/repositories/IImagesRepository';

import Image from '@modules/users/infra/typeorm/entities/Image';

@injectable()
class ListAllCasesOfAStudentService {
  constructor(
    @inject('UsersRepository')
    private usersRepository: IUsersRepository,

    @inject('ImagesRepository')
    private imagesRepository: IImagesRepository,
  ) {}

  public async execute(user_id: string): Promise<Image[]> {
    const checkUserExists = await this.usersRepository.findById(user_id);

    if (!checkUserExists) {
      throw new AppError('User does not exist');
    }

    const images = await this.imagesRepository.findAllImages(user_id);

    return images;
  }
}

export default ListAllCasesOfAStudentService;
