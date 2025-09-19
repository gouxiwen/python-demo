
-- DROP TABLE IF EXISTS user;
-- DROP TABLE IF EXISTS post;
-- DROP TABLE IF EXISTS post_like;
DROP TABLE IF EXISTS post_comment;


CREATE TABLE IF NOT EXISTS user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS post (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT (datetime(CURRENT_TIMESTAMP,'localtime')),
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  FOREIGN KEY (author_id) REFERENCES user (id)
);

-- 文章点赞表：每条记录表示某用户对某文章点赞
CREATE TABLE IF NOT EXISTS post_like (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  post_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT (datetime(CURRENT_TIMESTAMP,'localtime')),
  FOREIGN KEY (user_id) REFERENCES user (id),
  FOREIGN KEY (post_id) REFERENCES post (id),
  UNIQUE (user_id, post_id) -- 每个用户对每篇文章只能点赞一次
);

-- 文章评论表：每条记录表示某用户对某文章的评论
CREATE TABLE IF NOT EXISTS post_comment (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  username TEXT NOT NULL,
  post_id INTEGER NOT NULL,
  body TEXT NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT (datetime(CURRENT_TIMESTAMP,'localtime')),
  FOREIGN KEY (user_id) REFERENCES user (id),
  FOREIGN KEY (username) REFERENCES user (username),
  FOREIGN KEY (post_id) REFERENCES post (id)
);