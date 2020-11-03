import User from '../infra/typeorm/entities/User';
import ICreateUserDTO from '../dtos/ICreateUserDTO';

export default interface IMediasRepository {
  findById(id: string): Promise<User | undefined>;
  findByEmail(email: string): Promise<User | undefined>;
  checkIfExistByEmails(email: string): Promise<Boolean>;
  create(data: ICreateUserDTO): Promise<User>;
  save(user: User): Promise<User>;
  remove(user_id: string): Promise<void>;
}
