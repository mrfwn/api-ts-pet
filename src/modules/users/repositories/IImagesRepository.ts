import ICreateImageDTO from '../dtos/ICreateImageDTO';
import Image from '../infra/typeorm/entities/Image';

export default interface IFilesRepository {
  findAllImages(user_id: string): Promise<Image[]>;
  insertImages(images: ICreateImageDTO[]): Promise<Image[]>;
  insertPet(pet: ICreateImageDTO): Promise<Image>;
  deleteImages(imagesIds: string[]): Promise<void>;
}
