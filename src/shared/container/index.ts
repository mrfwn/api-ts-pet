import { container } from 'tsyringe';
import './providers';

import IUsersRepository from '@modules/users/repositories/IUsersRepository';
import UsersRepository from '@modules/users/infra/typeorm/repositories/UsersRepository';

import IImagesRepository from '@modules/users/repositories/IImagesRepository';
import ImagesRepository from '@modules/users/infra/typeorm/repositories/ImagesRepository';

container.registerSingleton<IUsersRepository>(
  'UsersRepository',
  UsersRepository,
);

container.registerSingleton<IImagesRepository>(
  'ImagesRepository',
  ImagesRepository,
);
