import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  ManyToOne,
  JoinColumn,
} from 'typeorm';

import uploadCofig from '@config/upload';
import { Expose } from 'class-transformer';
import User from './User';

@Entity('images')
class Image {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  user_id: string;

  @ManyToOne(() => User, user => user.images)
  @JoinColumn({ name: 'user_id' })
  user: User;

  @Column()
  name: string;

  @Column()
  type: boolean;

  @CreateDateColumn()
  created_at: Date;

  @UpdateDateColumn()
  updated_at: Date;

  @Expose({ name: 'path_url' })
  getAvatarUrl(): string | null {
    if (!this.name) {
      return null;
    }
    switch (uploadCofig.driver) {
      case 'disk':
        return `${process.env.APP_API_URL}/images/${this.name}`;
      default:
        return null;
    }
  }
}

export default Image;
