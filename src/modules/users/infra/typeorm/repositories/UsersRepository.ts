import { getRepository, Repository } from 'typeorm';
import IUsersRepository from '@modules/users/repositories/IUsersRepository';
import ICreateUserDTO from '@modules/users/dtos/ICreateUserDTO';

import User from '../entities/User';

class UsersRepository implements IUsersRepository {
  private ormRepository: Repository<User>;

  constructor() {
    this.ormRepository = getRepository(User);
  }

  public async findById(id: string): Promise<User | undefined> {
    const user = await this.ormRepository.findOne(id);
    return user;
  }

  public async findByEmail(email: string): Promise<User | undefined> {
    const user = await this.ormRepository.findOne({
      where: { email },
    });
    return user;
  }

  public async checkIfExistByEmails(email: string): Promise<Boolean> {
    const user = await this.ormRepository.findOne({
      where: { email },
    });

    return !!user;
  }

  public async create(UserData: ICreateUserDTO): Promise<User> {
    const user = this.ormRepository.create(UserData);

    await this.ormRepository.save(user);

    return user;
  }

  public async save(user: User): Promise<User> {
    return this.ormRepository.save(user);
  }

  public async remove(user_id: string): Promise<void> {
    const user = await this.ormRepository.findOne(user_id);
    if (user) {
      await this.ormRepository.remove(user);
    }
  }
}

export default UsersRepository;
