from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.auth import login_required
from flaskr.db import get_db

bp = Blueprint('blog', __name__)


@bp.route('/')
def index():
    query = request.args.get('query', '')
    page = request.args.get('page', 1, type=int)
    per_page = 5  # 每页显示5条数据

    db = get_db()

    # 获取总记录数
    total_count = db.execute(
        'SELECT COUNT(*) FROM post'
        ' WHERE title LIKE ? OR body LIKE ?',
        ('%' + query + '%', '%' + query + '%')
    ).fetchone()[0]

    # 计算总页数
    total_pages = (total_count + per_page - 1) // per_page

    # 获取当前页的数据
    offset = (page - 1) * per_page
    posts = db.execute(
        'SELECT p.id, title, body, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' WHERE p.title LIKE ? OR p.body LIKE ?'
        ' ORDER BY created DESC'
        ' LIMIT ? OFFSET ?',
        ('%' + query + '%', '%' + query + '%', per_page, offset)
    ).fetchall()

    # 查询所有标签
    tags = get_tags()
    return render_template('blog/index.html', 
                         posts=posts, 
                         query=query, 
                         tags=tags,
                         page=page,
                         total_pages=total_pages,
                         current_url=request.url)

# 通过标签查询文章列表
@bp.route('/tag/<int:tag_id>')
def tag_posts(tag_id):
    db = get_db()
    tag = db.execute('SELECT id, name FROM tag WHERE id = ?', (tag_id,)).fetchone()
    if not tag:
        abort(404, '标签不存在')

    page = request.args.get('page', 1, type=int)
    per_page = 5  # 每页显示5条数据

    # 获取总记录数
    total_count = db.execute(
        'SELECT COUNT(*) FROM post p'
        ' JOIN post_tag pt ON pt.post_id = p.id'
        ' WHERE pt.tag_id = ?',
        (tag_id,)
    ).fetchone()[0]

    # 计算总页数
    total_pages = (total_count + per_page - 1) // per_page

    # 获取当前页的数据
    offset = (page - 1) * per_page
    posts = db.execute(
        'SELECT p.id, title, body, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' JOIN post_tag pt ON pt.post_id = p.id'
        ' WHERE pt.tag_id = ?'
        ' ORDER BY created DESC'
        ' LIMIT ? OFFSET ?',
        (tag_id, per_page, offset)
    ).fetchall()

    # 查询所有标签
    tags = get_tags()
    return render_template('blog/index.html', 
                         posts=posts, 
                         query='', 
                         tags=tags, 
                         current_tag=tag,
                         page=page,
                         total_pages=total_pages,
                         current_url=request.url)

def get_tags_by_post(post_id):
    db = get_db()
    tags = db.execute(
        'SELECT t.id, t.name FROM tag t'
        ' JOIN post_tag pt ON t.id = pt.tag_id'
        ' WHERE pt.post_id = ?',
        (post_id,)
    ).fetchall()
    return tags

def get_tags():
    db = get_db()
    tags = db.execute('SELECT id, name FROM tag ORDER BY name').fetchall()
    return tags

@bp.route('/create', methods=('GET', 'POST'))
@login_required
def create():
    db = get_db()
    tags = db.execute('SELECT id, name FROM tag ORDER BY name').fetchall()
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        tag_ids = request.form.getlist('tags')
        error = None

        if not title:
            error = 'Title is required.'

        if error is not None:
            flash(error)
        else:
            cursor = db.execute(
                'INSERT INTO post (title, body, author_id)'
                ' VALUES (?, ?, ?)',
                (title, body, g.user['id'])
            )
            post_id = cursor.lastrowid
            # 关联标签
            for tag_id in tag_ids:
                db.execute('INSERT INTO post_tag (post_id, tag_id) VALUES (?, ?)', (post_id, tag_id))
            db.commit()
            return redirect(url_for('blog.index'))

    return render_template('blog/create.html', tags=tags)

def get_post(id, check_author=True):
    post = get_db().execute(
        'SELECT p.id, title, body, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' WHERE p.id = ?',
        (id,)
    ).fetchone()

    if post is None:
        abort(404, f"Post id {id} doesn't exist.")

    if check_author and post['author_id'] != g.user['id']:
        abort(403)

    return post

@bp.route('/<int:id>/update', methods=('GET', 'POST'))
@login_required
def update(id):
    post = get_post(id)
    tags = get_tags()
    post_tags = get_tags_by_post(id)

    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        tag_ids = request.form.getlist('tags')
        error = None

        if not title:
            error = 'Title is required.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            # 更新文章内容
            db.execute(
                'UPDATE post SET title = ?, body = ?'
                ' WHERE id = ?',
                (title, body, id)
            )

            # 删除原有的标签关联
            db.execute('DELETE FROM post_tag WHERE post_id = ?', (id,))

            # 添加新的标签关联
            for tag_id in tag_ids:
                db.execute('INSERT INTO post_tag (post_id, tag_id) VALUES (?, ?)', (id, tag_id))

            db.commit()
            return redirect(url_for('blog.index'))

    return render_template('blog/update.html', post=post, tags=tags, post_tags=[pt['id'] for pt in post_tags])

@bp.route('/<int:id>/view', methods=('GET',))
def view(id):
    post = get_post(id, False)
    likes = get_post_likes(id)
    comments = get_post_comments(id)
    tags = get_tags_by_post(id)
    if g.user:
        post_liked_by_user = get_post_liked_by_user(id)
    else:
        post_liked_by_user = 0
    return render_template('blog/view.html', post=post, likes=likes, post_liked_by_user=post_liked_by_user, comments=comments, tags=tags)

@bp.route('/<int:id>/delete', methods=('POST',))
@login_required
def delete(id):
    get_post(id)
    db = get_db()
    db.execute('DELETE FROM post WHERE id = ?', (id,))
    db.commit()
    return redirect(url_for('blog.index'))

# 点赞功能
@bp.route('/<int:id>/<string:like>/like', methods=('GET',))
@login_required
def like(id, like):
    post = get_post(id)
    db = get_db()
    if like == 'True':
        db.execute('INSERT INTO post_like (user_id, post_id) VALUES (?, ?)',
                   (g.user['id'], post['id']))
    elif like == 'False':
        db.execute('DELETE FROM post_like WHERE user_id = ? AND post_id = ?',
                   (g.user['id'], post['id']))
    db.commit()
    return redirect(url_for('blog.view', id=post['id']))

def get_post_likes(id):
    post_like_count = get_db().execute(
        'SELECT COUNT(*) FROM post_like WHERE post_id = ?',
        (id,)
    ).fetchone()
    return post_like_count[0] if post_like_count else 0

def get_post_liked_by_user(id):
    liked = get_db().execute(
        'SELECT COUNT(*) FROM post_like WHERE post_id = ? AND user_id = ?',
        (id, g.user['id'])
    ).fetchone()
    return liked[0] if liked else 0

# 评论功能
def get_post_comments(id):
    comments = get_db().execute(
        'SELECT c.id, body, created, user_id, username'
        ' FROM post_comment c'
        ' WHERE c.post_id = ?'
        ' ORDER BY created DESC',
        (id,)
    ).fetchall()
    return comments

@bp.route('/<int:id>/add_comment', methods=('POST',))
@login_required
def add_comment(id):
    body = request.form['body']
    db = get_db()
    db.execute(
        'INSERT INTO post_comment (body, user_id, username, post_id)'
        ' VALUES (?, ?, ?, ?)',
        (body, g.user['id'], g.user['username'], id)
    )
    db.commit()
    return redirect(url_for('blog.view', id=id))

@bp.route('/<int:id>/delete_comment', methods=('POST',))
@login_required
def delete_comment(id):
    db = get_db()
    db.execute('DELETE FROM post_comment WHERE id = ? AND user_id = ?', (id, g.user['id']))
    db.commit()
    return {"code": 200, "message": "Comment deleted"}

