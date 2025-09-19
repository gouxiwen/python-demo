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
    db = get_db()
    posts = db.execute(
        'SELECT p.id, title, body, created, author_id, username'
        ' FROM post p JOIN user u ON p.author_id = u.id'
        ' WHERE p.title LIKE ? OR p.body LIKE ?'
        ' ORDER BY created DESC',
        ('%' + query + '%', '%' + query + '%')
    ).fetchall()
    return render_template('blog/index.html', posts=posts, query=query)

@bp.route('/create', methods=('GET', 'POST'))
@login_required
def create():
    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        error = None

        if not title:
            error = 'Title is required.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'INSERT INTO post (title, body, author_id)'
                ' VALUES (?, ?, ?)',
                (title, body, g.user['id'])
            )
            db.commit()
            return redirect(url_for('blog.index'))

    return render_template('blog/create.html')

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

    if request.method == 'POST':
        title = request.form['title']
        body = request.form['body']
        error = None

        if not title:
            error = 'Title is required.'

        if error is not None:
            flash(error)
        else:
            db = get_db()
            db.execute(
                'UPDATE post SET title = ?, body = ?'
                ' WHERE id = ?',
                (title, body, id)
            )
            db.commit()
            return redirect(url_for('blog.index'))

    return render_template('blog/update.html', post=post)

@bp.route('/<int:id>/view', methods=('GET',))
def view(id):
    post = get_post(id, False)
    likes = get_post_likes(id)
    comments = get_post_comments(id)
    if g.user:
        post_liked_by_user = get_post_liked_by_user(id)
    else:
        post_liked_by_user = 0
    return render_template('blog/view.html', post=post, likes=likes, post_liked_by_user=post_liked_by_user, comments=comments)

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